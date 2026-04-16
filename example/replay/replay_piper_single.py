import argparse
import time
import os
import h5py
import numpy as np
import sys
sys.path.append('/home/charles/workspaces/Double_Piper_Teleop')
from my_robot.agilex_piper_single_base import PiperSingle

# OpenCV 检测
_CV2_AVAILABLE = False
_CV2_GUI_AVAILABLE = False
try:
    import cv2
    _CV2_AVAILABLE = True
    try:
        cv2.namedWindow("__cv2_test__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__cv2_test__")
        _CV2_GUI_AVAILABLE = True
    except Exception:
        _CV2_GUI_AVAILABLE = False
except Exception:
    _CV2_AVAILABLE = False
    _CV2_GUI_AVAILABLE = False

# Tkinter + PIL 作为 OpenCV GUI 的备用（在 opencv 无 GUI 后端时使用）
_TK_AVAILABLE = False
try:
    import tkinter as tk
    from PIL import Image, ImageTk
    _TK_AVAILABLE = True
except Exception:
    _TK_AVAILABLE = False

def print_h5_tree(h5obj, prefix=""):
    for k in h5obj:
        obj = h5obj[k]
        path = f"{prefix}/{k}" if prefix else k
        if isinstance(obj, h5py.Dataset):
            print(f"D: {path}  shape={obj.shape}  dtype={obj.dtype}")
        else:
            print(f"G: {path}")
            print_h5_tree(obj, path)

def find_datasets(h5f):
    ds_paths = []
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            ds_paths.append(name)
    h5f.visititems(visitor)

    joint = None
    gripper = None
    time_ds = None
    image_datasets = []
    for name in ds_paths:
        lname = name.lower()
        if "joint" in lname and joint is None:
            joint = name
        if "gripper" in lname and gripper is None:
            gripper = name
        if ("time" in lname or "timestamp" in lname) and time_ds is None:
            time_ds = name
        if "image" in lname or "color" in lname:
            image_datasets.append(name)
    return joint, gripper, time_ds, image_datasets, ds_paths

def load_array(h5f, path):
    arr = h5f[path][()]
    return np.asarray(arr)

def normalize_image_frame(arr):
    img = np.asarray(arr)
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = np.clip(img, 0.0, 1.0)
        img = (img * 255.0).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    # 目标输出：H x W x 3 uint8
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = np.concatenate([img]*3, axis=2)
    elif img.ndim == 3 and img.shape[2] == 3:
        pass
    else:
        img = np.zeros((240, 320, 3), dtype=np.uint8)
    return img

def _init_tk_windows(n_images, win_names):
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    tops = []
    labels = []
    for wn in win_names:
        top = tk.Toplevel(root)
        top.title(wn)
        lbl = tk.Label(top)
        lbl.pack()
        tops.append(top)
        labels.append(lbl)
    return root, tops, labels

def replay_file(hdf5_path, can_name="piper_slave", rate=30.0, compare=False, save_images=False, save_dir="/tmp/replay_frames"):
    p = PiperSingle()
    p.set_can_name(can_name)
    p.set_up()
    p.reset()

    print(f"[DIAG] _CV2_AVAILABLE={_CV2_AVAILABLE}, _CV2_GUI_AVAILABLE={_CV2_GUI_AVAILABLE}, TK_AVAILABLE={_TK_AVAILABLE}, DISPLAY={os.environ.get('DISPLAY')}")
    if save_images:
        os.makedirs(save_dir, exist_ok=True)
        print(f"[DIAG] saving frames to: {save_dir}")

    gui_mode = None
    if _CV2_AVAILABLE and _CV2_GUI_AVAILABLE:
        gui_mode = "cv2"
    elif _TK_AVAILABLE:
        gui_mode = "tk"
    else:
        gui_mode = None

    with h5py.File(hdf5_path, "r") as f:
        print("HDF5 content:")
        print_h5_tree(f)
        joint_ds, grip_ds, time_ds, image_ds_list, all_ds = find_datasets(f)
        if joint_ds is None or grip_ds is None:
            raise RuntimeError(f"未找到 joint/gripper 数据集。候选数据集: {all_ds}")
        joints = load_array(f, joint_ds)
        grippers = load_array(f, grip_ds)

        # 选择最多两路图像，优先 wrist，其次 head/其他
        images_arrays = []
        chosen_names = []
        preferred = []
        for name in image_ds_list:
            ln = name.lower()
            if "wrist" in ln:
                preferred.insert(0, name)
            else:
                preferred.append(name)
        for name in preferred[:2]:
            try:
                arr = load_array(f, name)
                images_arrays.append(arr)
                chosen_names.append(name)
            except Exception as e:
                print(f"[WARN] 无法加载图像数据集 {name}: {e}")

        # 对齐帧数（以最短为准）
        lengths = [getattr(joints, "shape", (len(joints),))[0], getattr(grippers, "shape", (len(grippers),))[0]]
        for arr in images_arrays:
            lengths.append(getattr(arr, "shape", (len(arr),))[0])
        n = int(min(lengths))
        joints = joints[:n]
        grippers = grippers[:n]
        images_arrays = [arr[:n] for arr in images_arrays]

        period = 1.0 / float(rate) if rate > 0 else 0.0
        print(f"Replaying {n} frames at ~{rate}Hz (period {period:.4f}s)")

        # 确定窗口名（固定：image, wrist_image）
        win_names = []
        for name in chosen_names:
            ln = name.lower()
            if "wrist" in ln:
                win_names.append("wrist_image")
            elif "head" in ln or "cam_head" in ln:
                win_names.append("image")
            else:
                if "image" not in win_names:
                    win_names.append("image")
                else:
                    win_names.append(f"image_{len(win_names)}")

        # 创建 GUI 窗口（cv2 or tk）
        tk_root = None
        tk_labels = None
        if gui_mode == "cv2":
            for wn in win_names:
                try:
                    cv2.namedWindow(wn, cv2.WINDOW_NORMAL)
                except Exception as e:
                    print(f"[WARN] cv2.namedWindow 失败: {e}")
                    gui_mode = None
                    break
        elif gui_mode == "tk":
            try:
                tk_root, _, tk_labels = _init_tk_windows(len(win_names), win_names)
            except Exception as e:
                print(f"[WARN] tkinter 创建窗口失败: {e}")
                gui_mode = None

        try:
            for i in range(n):
                joint_frame = np.asarray(joints[i]).astype(float).flatten()
                grip_val = float(np.asarray(grippers[i]).squeeze())

                # # 安全限制：每个 joint 绝对值不得超过 0.1
                # clipped = np.clip(joint_frame, -0.1, 0.1)
                # if not np.allclose(clipped, joint_frame):
                #     print(f"[WARN] frame {i}: joint values clipped to ±0.1")
                # joint_frame = clipped

                move = {"joint": [float(x) for x in joint_frame], "gripper": float(grip_val)}
                # 发动作（如需实际执行则取消下一行注释）
                p.move({"arm": {"left_arm": move}})

                if compare:
                    try:
                        sensor = p.get()
                        print(f"[{i}] sent joint={np.round(joint_frame,3).tolist()} gripper={grip_val}")
                        print(f"     sensor sample: {sensor}")
                    except Exception as e:
                        print(f"[{i}] sent (compare failed: {e})")

                # 显示并/或保存对应帧图像
                for idx, arr in enumerate(images_arrays):
                    try:
                        frame = arr[i]
                        img = cv2.cvtColor(normalize_image_frame(frame), cv2.COLOR_RGB2BGR) 
                        win = win_names[idx] if idx < len(win_names) else f"image_{idx}"

                        if gui_mode == "cv2":
                            try:
                                cv2.imshow(win, img)
                            except Exception as e:
                                print(f"[WARN] cv2.imshow 失败 idx={idx} i={i}: {e}")
                                gui_mode = None

                        elif gui_mode == "tk":
                            try:
                                pil = Image.fromarray(img)
                                imgtk = ImageTk.PhotoImage(pil)
                                lbl = tk_labels[idx]
                                lbl.imgtk = imgtk
                                lbl.config(image=imgtk)
                            except Exception as e:
                                print(f"[WARN] tkinter 更新失败 idx={idx} i={i}: {e}")
                                gui_mode = None

                        if save_images:
                            fname = os.path.join(save_dir, f"{i:04d}_{win}.png")
                            try:
                                if _CV2_AVAILABLE:
                                    cv2.imwrite(fname, img)
                                else:
                                    from PIL import Image as PILImage
                                    PILImage.fromarray(img).save(fname)
                            except Exception as e:
                                print(f"[WARN] 无法保存帧 {i} idx={idx}: {e}")
                    except Exception as e:
                        print(f"[WARN] 处理图像帧失败 idx={idx} i={i}: {e}")

                # 处理 GUI 事件
                if gui_mode == "cv2":
                    try:
                        cv2.waitKey(1)
                    except Exception:
                        gui_mode = None
                elif gui_mode == "tk":
                    try:
                        tk_root.update_idletasks()
                        tk_root.update()
                    except Exception:
                        gui_mode = None

                if period > 0:
                    time.sleep(period)
        except KeyboardInterrupt:
            print("Replay interrupted by user.")
        finally:
            if gui_mode == "cv2":
                try:
                    cv2.destroyAllWindows()
                except Exception as e:
                    print(f"[WARN] cv2.destroyAllWindows 失败: {e}")
            elif gui_mode == "tk":
                try:
                    tk_root.destroy()
                except Exception:
                    pass
    print("Replay finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从 hdf5 回放动作到 PiperSingle（含同步图像回放）")
    parser.add_argument("hdf5", nargs="?", default=os.path.join(os.path.dirname(__file__), "/home/charles/workspaces/Double_Piper_Teleop/datasets/pick_banana_200_noop/0041.hdf5"),
                        help="hdf5 文件路径（默认 datasets/test/0.hdf5）")
    parser.add_argument("--can", default="can0", help="CAN 名称，传给 PiperSingle.set_can_name")
    parser.add_argument("--rate", type=float, default=10.0, help="回放频率 (Hz)")
    parser.add_argument("--compare", action="store_true", help="同时读取当前控制器返回的传感器数据并打印，便于对比")
    parser.add_argument("--save-images", action="store_true", help="保存每帧图像到磁盘（便于 headless 环境调试）")
    parser.add_argument("--save-dir", default="/tmp/replay_frames", help="保存图像的目录（配合 --save-images）")
    args = parser.parse_args()

    replay_file(os.path.abspath(args.hdf5), can_name=args.can, rate=args.rate, compare=args.compare,
                save_images=args.save_images, save_dir=args.save_dir)