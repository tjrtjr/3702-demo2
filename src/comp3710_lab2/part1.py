import time

import matplotlib.pyplot as plt
import numpy as np
import torch


# 讲义原代码给出的信号参数
N = 2048
T = 1.0
f0 = 1
# 讲义先比较 1、3、5 项，作业要求继续比较 20、50 项
harmonics = [1, 3, 5, 20, 50]
has_cuda = torch.cuda.is_available()


# 下面三个 NumPy 核心函数直接复制自讲义，只把英文注释改成中文
def square_wave(t):
    return np.sign(np.sin(2.0 * np.pi * f0 * t))


def square_wave_fourier(t, f0, N):
    result = np.zeros_like(t)
    for k in range(N):
        n = 2 * k + 1  # 方波只包含奇次谐波
        result += np.sin(2 * np.pi * n * f0 * t) / n
    return (4 / np.pi) * result


def naive_dft(x):
    N = len(x)
    X = np.zeros(N, dtype=np.complex128)
    # 按 DFT 公式逐项计算，因此时间复杂度为 O(N²)
    for k in range(N):
        for n in range(N):
            angle = -2j * np.pi * k * n / N
            X[k] += x[n] * np.exp(angle)
    return X


# 用 PyTorch 运算重写讲义中的三个函数
def square_wave_torch(t):
    return torch.sign(torch.sin(2.0 * torch.pi * f0 * t))


def square_wave_fourier_torch(t, f0, N):
    result = torch.zeros_like(t)
    for k in range(N):
        n = 2 * k + 1
        result += torch.sin(2 * torch.pi * n * f0 * t) / n
    return (4 / torch.pi) * result


def naive_dft_torch(x):
    N = x.numel()
    n = torch.arange(N, device=x.device, dtype=x.dtype)
    k = n[:, None]
    # 一次构造所有频率与采样点的指数项，再用矩阵乘法并行求和
    matrix = torch.exp(-2j * torch.pi * k * n / N)
    return matrix @ x.to(matrix.dtype)


def elapsed(function, x, repeats=1, gpu=False):
    if gpu:
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        function(x)
    if gpu:
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats


# 1. 比较不同项数的傅里叶级数近似
t = np.linspace(0.0, T, N, endpoint=False)
square = square_wave(t)

plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1)
plt.plot(t, square, "k", label="Square wave")
plt.title("Original Square Wave")
plt.ylim(-1.5, 1.5)
plt.grid(True)
plt.legend()

for i, Nh in enumerate(harmonics, start=2):
    plt.subplot(2, 3, i)
    y = square_wave_fourier(t, f0, Nh)
    plt.plot(t, y, label=f"N={Nh} harmonics")
    plt.plot(t, square, "k--", alpha=0.5, label="Square wave")
    plt.title(f"Fourier Approximation with N={Nh}")
    plt.ylim(-1.5, 1.5)
    plt.grid(True)
    plt.legend()

# 项数越多，边缘越陡；但跳变处仍有 Gibbs 过冲，只是振荡区域变窄
plt.tight_layout()


# 2. 按讲义比较朴素 DFT 与 NumPy FFT
signal = square_wave_fourier(t, f0, 50)

start = time.perf_counter()
dft_result = naive_dft(signal)
naive_time = time.perf_counter() - start

start = time.perf_counter()
for _ in range(100):
    fft_result = np.fft.fft(signal)
fft_time = (time.perf_counter() - start) / 100

print("--- DFT/FFT 正确性与速度 ---")
print(f"朴素 DFT: {naive_time:.6f} 秒")
print(f"NumPy FFT: {fft_time:.6f} 秒")
print(f"结果一致: {np.allclose(dft_result, fft_result)}")

xf = np.fft.fftfreq(N, d=T / N)[: N // 2]
magnitude = 2.0 / N * np.abs(dft_result[: N // 2])

plt.style.use("seaborn-v0_8-darkgrid")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
ax1.plot(t, signal, color="c")
ax1.set_title("Input Square Wave Signal")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Amplitude")

ax2.stem(xf, magnitude, basefmt="")
ax2.set_title("Discrete Fourier Transform (Magnitude Spectrum)")
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Magnitude")
# 按讲义显示 0–50 Hz，并标出前十个奇次谐波。
# 50 项实际包含 1、3、5、…、99 Hz，超出显示范围的分量仍在 DFT 结果中。
ax2.set_xlim(0, 50)
for i in range(20):
    if i < len(xf) and i % 2 == 1:
        ax2.axvline(
            xf[i], color="r", linestyle="--", alpha=0.7,
            label=f"f{i}: {i}*f0 = {xf[i]:.1f} Hz",
        )
ax2.legend()

plt.tight_layout()


# 3. 检查 PyTorch 改写结果
t_torch = torch.arange(N, dtype=torch.float64) * T / N
square_torch = square_wave_torch(t_torch)
signal_torch = square_wave_fourier_torch(t_torch, f0, 50)
dft_torch = naive_dft_torch(signal_torch)

print("\n--- PyTorch 改写检查 ---")
print(f"方波一致: {np.allclose(square_torch.numpy(), square)}")
print(f"傅里叶近似一致: {np.allclose(signal_torch.numpy(), signal)}")
print(f"PyTorch 朴素 DFT 一致: {np.allclose(dft_torch.numpy(), fft_result)}")
if has_cuda:
    dft_gpu = naive_dft_torch(signal_torch.to("cuda"))
    torch.cuda.synchronize()
    print(f"GPU 设备: {dft_gpu.device}")
    print(f"GPU 朴素 DFT 一致: {np.allclose(dft_gpu.cpu().numpy(), fft_result)}")


# 4. 比较作业指定的三种方法：NumPy 朴素 DFT、NumPy FFT、PyTorch GPU 朴素 DFT
sizes = [256, 512, 1024, 2048]
timings = []

if has_cuda:
    naive_dft_torch(torch.ones(32, dtype=torch.float64, device="cuda"))  # GPU 预热
    torch.cuda.synchronize()

for size in sizes:
    time_points = np.linspace(0.0, T, size, endpoint=False)
    x_numpy = square_wave_fourier(time_points, f0, 50)
    x_torch = torch.from_numpy(x_numpy)

    numpy_naive = elapsed(naive_dft, x_numpy)
    numpy_fft = elapsed(np.fft.fft, x_numpy, repeats=100)
    # 输入先放到 GPU，因此计时只包含计算；小 N 时启动和同步开销可能抵消并行优势
    x_gpu = x_torch.to("cuda") if has_cuda else None
    torch_gpu = elapsed(naive_dft_torch, x_gpu, repeats=3, gpu=True) if has_cuda else None
    timings.append([size, numpy_naive, numpy_fft, torch_gpu])

print("\n--- 不同数据大小的运行时间（秒）---")
print(f"{'N':>6} {'NumPy 朴素':>14} {'NumPy FFT':>14} {'PyTorch GPU':>14}")
for size, numpy_naive, numpy_fft, torch_gpu in timings:
    gpu_text = f"{torch_gpu:.6f}" if torch_gpu is not None else "待 Rangpur"
    print(f"{size:>6} {numpy_naive:>14.6f} {numpy_fft:>14.6f} {gpu_text:>14}")

names = ["NumPy 朴素 DFT", "NumPy FFT"]
times = timings[-1][1:3]
if timings[-1][3] is not None:
    names.append("PyTorch 朴素 DFT（GPU）")
    times.append(timings[-1][3])
# FFT 是 O(N log N)；两个朴素 DFT 是 O(N²)，GPU 版本依靠张量并行加速
order = [name for _, name in sorted(zip(times, names))]
print("N=2048 从快到慢：" + " < ".join(order))

if not has_cuda:
    print("本机没有 CUDA；GPU 一列需要在 Rangpur 上运行同一脚本补齐。")

plt.show()
