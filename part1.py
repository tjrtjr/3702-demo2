import time

import matplotlib.pyplot as plt
import numpy as np
import torch


# 讲义给出的信号参数
N = 2048
T = 1.0
f0 = 1
harmonics = [1, 3, 5, 20, 50]
has_cuda = torch.cuda.is_available()


# 以下三个 NumPy 函数沿用讲义写法
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

ax2.plot(xf, magnitude, color="m")
ax2.set_title("Discrete Fourier Transform (Magnitude Spectrum)")
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Magnitude")
ax2.set_xlim(0, 105)  # 50 项对应 1 到 99 Hz 的奇次谐波
expected_frequencies = np.arange(1, 100, 2)
ax2.scatter(expected_frequencies, magnitude[expected_frequencies], s=15, label="Odd harmonics")
ax2.legend()

plt.tight_layout()

print("频谱峰值位于 1、3、5、…、99 Hz，与构造方波使用的 50 个奇次谐波一致。")
print("其他位置的微小数值来自有限采样和浮点误差；有限项也无法得到完全理想的方波。")


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


# 4. 改变数据大小并比较计算时间
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
    torch_cpu = elapsed(naive_dft_torch, x_torch)
    x_gpu = x_torch.to("cuda") if has_cuda else None
    torch_gpu = elapsed(naive_dft_torch, x_gpu, repeats=3, gpu=True) if has_cuda else None
    timings.append([size, numpy_naive, numpy_fft, torch_cpu, torch_gpu])

print("\n--- 不同数据大小的运行时间（秒）---")
if has_cuda:
    print("GPU 计时只包含计算，输入在计时前已经放到显卡。")
print(f"{'N':>6} {'NumPy 朴素':>14} {'NumPy FFT':>14} {'PyTorch CPU':>14} {'PyTorch GPU':>14}")
for size, numpy_naive, numpy_fft, torch_cpu, torch_gpu in timings:
    gpu_text = f"{torch_gpu:.6f}" if torch_gpu is not None else "待 Rangpur"
    print(f"{size:>6} {numpy_naive:>14.6f} {numpy_fft:>14.6f} {torch_cpu:>14.6f} {gpu_text:>14}")

names = ["NumPy 朴素 DFT", "NumPy FFT", "PyTorch 朴素 DFT（CPU）"]
times = timings[-1][1:4]
if timings[-1][4] is not None:
    names.append("PyTorch 朴素 DFT（GPU）")
    times.append(timings[-1][4])
order = [name for _, name in sorted(zip(times, names))]
print("N=2048 从快到慢：" + " < ".join(order))

plt.figure(figsize=(8, 5))
plt.loglog(sizes, [row[1] for row in timings], "o-", label="NumPy naive DFT")
plt.loglog(sizes, [row[2] for row in timings], "o-", label="NumPy FFT")
plt.loglog(sizes, [row[3] for row in timings], "o-", label="PyTorch naive DFT (CPU)")
if has_cuda:
    plt.loglog(sizes, [row[4] for row in timings], "o-", label="PyTorch naive DFT (GPU)")
plt.xlabel("N")
plt.ylabel("Time (s)")
plt.title("DFT Timing Comparison")
plt.legend()
plt.tight_layout()

print("\n说明：增加奇次谐波会让方波边缘更陡，但跳变处的 Gibbs 过冲不会消失。")
print("朴素 DFT 是 O(N²)，FFT 约为 O(N log N)，所以数据越大，FFT 的优势越明显。")
print("PyTorch 用张量矩阵运算代替 Python 双循环；GPU 可以并行，但小 N 的启动和同步开销仍可能占主导。")
if not has_cuda:
    print("本机没有 CUDA；GPU 一列需要在 Rangpur 上运行同一脚本补齐。")

plt.show()
