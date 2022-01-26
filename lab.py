from math import floor
import numpy as np
import random as rd
import matplotlib.pyplot  as plt
from scipy import signal, fftpack

def test():
    bit_signal = get_random_bits(100000) # prueba de 10**5
    f0, f1, t, modulated_signal = modulation(bit_signal, bit_rate, f_s)
    demodulation(f0, f1, t, bit_signal, modulated_signal, bit_rate, f_s)
    return None

def get_fourier_comps(signal, f_s):
    fft = fftpack.fft(signal)
    freqs = fftpack.fftfreq(len(fft), d = 1/f_s)
    return freqs, fft

def get_random_bits(n):
    return [rd.randint(0, 1) for i in range(n)]

def modulation(bit_signal, bit_rate, f_s):
    bit_time = 1/bit_rate
    f0 = bit_rate*3
    f1 = bit_rate*6
    
    endpoint = 1/f_s
    t_c = np.arange(0, bit_time + endpoint, endpoint)

    carrier0 = np.cos(2*np.pi*f0*t_c)
    carrier1 = np.cos(2*np.pi*f1*t_c)

    y = []
    for bit in bit_signal:
        if bit == 0:
            y.extend(carrier0)
        else:
            y.extend(carrier1)

    total_time = np.linspace(0, bit_time*len(bit_signal), len(y))

    # plt.plot(total_time, y)
    # plt.show()
    # plt.clf()

    return f0, f1, total_time, y

def demodulation(f0, f1, t, bit_signal, modulated_signal, bit_rate, f_s):
    """Función que demodula una señal de entrada FSK.
    Args:
        f0 (int): freq. de portadora para símbolo 0
        f1 (int): freq. portadora símbolo 1
        t (array): vector temporal, en segundos
        bit_signal (array): cadena de binarios original
        modulated_signal (array): señal modulada con FSK
        bit_rate (int): bit rate en bps
        f_s (int): sampling rate de señales, >> a frecuencias portadoras
    Returns:
        ---: ---
    """
    bit_time = 1/bit_rate
    endpoint = 1/f_s
    t_c = np.arange(0, bit_time + endpoint, endpoint)
    c0 = np.cos(2*np.pi*f0*t_c)
    c1 = np.cos(2*np.pi*f1*t_c)

    # freqs, ft = get_fourier_comps(modulated_signal, f_s)
    # plt.plot(fftpack.fftshift(freqs), fftpack.fftshift(np.abs(ft)))
    # plt.show()
    # plt.clf()

    m = 'same' # parámetro por defecto solo devuelve un escalar
    s0 = np.correlate(modulated_signal, c0, mode = m)
    s1 = np.correlate(modulated_signal, c1, mode = m)

    s0 = signal.hilbert(np.gradient(s0))
    s0 = np.abs(s0)

    s1 = signal.hilbert(np.gradient(s1))
    s1 = np.abs(s1)


    # plt.subplot(3, 1, 1)
    # plt.plot(t, modulated_signal)

    # plt.subplot(3, 1, 2)
    # plt.plot(t, s0)

    # plt.subplot(3, 1, 3)
    # plt.plot(t, s1)
    # plt.show()

    out = []
    step = len(c0)
    i = floor(step/2)
    while i < len(modulated_signal):
        if s1[i] > s0[i]:
            out.append(1)
        else:
            out.append(0)
        i += step

    print(bit_signal)
    print(out)
    print("Resultado de demodulacion: ", out == bit_signal)

    return None


def awgn(SNR_rate, modulated_signal, t):
    p_signal = sum(np.abs(modulated_signal)**2)/(2*len(modulated_signal) + 1)
    linear_snr = 10**(SNR_rate/10)
    sigma = np.sqrt(p_signal/linear_snr)
    print(linear_snr)
    print(p_signal)
    print(sigma)

    rng = np.random.default_rng()
    noise = sigma*rng.standard_normal(len(modulated_signal))

    # plt.plot(t, noise)
    # plt.show()
    # plt.clf()
    out = np.add(modulated_signal, noise)

    return out

if __name__ == '__main__':
    bit_rate = 300 # 300 bps
    f_s = 22050 
    bit_signal = get_random_bits(5)
    print(bit_signal)
    f0, f1, t, modulated_signal = modulation(bit_signal, bit_rate, f_s)
    # demodulation(f0, f1, t, bit_signal, modulated_signal, bit_rate, f_s)

    noise_signal = awgn(3, modulated_signal, t) # 3 dB
    plt.subplot(2, 1, 1)
    plt.plot(t, modulated_signal)
    plt.subplot(2, 1, 2)
    plt.plot(t, noise_signal)
    plt.show()