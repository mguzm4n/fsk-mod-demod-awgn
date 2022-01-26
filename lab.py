from math import floor
from tabnanny import check
import numpy as np
import random as rd
import matplotlib.pyplot  as plt
from scipy import signal, fftpack
import time

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

def demodulation(f0, f1, t, modulated_signal, bit_rate, f_s):
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


    plt.subplot(3, 1, 1)
    plt.plot(t, modulated_signal)

    plt.subplot(3, 1, 2)
    plt.plot(t, s0)

    plt.subplot(3, 1, 3)
    plt.plot(t, s1)
    plt.show()

    out = []
    step = len(c0)
    i = floor(step/2)
    while i < len(modulated_signal):
        if s1[i] > s0[i]:
            out.append(1)
        else:
            out.append(0)
        i += step


    return out


def awgn(SNR_rate, modulated_signal, t):
    p_signal = sum(np.abs(modulated_signal)**2)/(2*len(modulated_signal) + 1)
    linear_snr = 10**(SNR_rate/10)
    sigma = np.sqrt(p_signal/linear_snr)

    # print(linear_snr)
    # print(p_signal)
    # print(sigma)

    rng = np.random.default_rng()
    noise = sigma*rng.standard_normal(len(modulated_signal))

    out = np.add(modulated_signal, noise)

    return out

def check_errors(original_bits, demod_bits):
    count = np.sum(np.array(original_bits) != np.array(demod_bits))
    return count/len(original_bits)


def plot_time(t, y, title, filename):
    plt.plot(t, y)
    plt.title(title)
    plt.show()
    #plt.savefig(filename + ".png")
    plt.clf()

if __name__ == '__main__':
    # parte 1, 2 3

    bit_rate = 300 # 300 bps
    f_s = 22050 
    bit_signal = [1, 1, 1, 0, 1, 0, 1, 0, 0, 1] # pequeña muestra
    f0, f1, t, modulated_signal = modulation(bit_signal, bit_rate, f_s)

    plot_time(t, modulated_signal, "Señal Modulada (largo pequeño)", "mod10")

    out_test = demodulation(f0, f1, t, modulated_signal, bit_rate, f_s)

    if out_test == bit_signal: # Se prueba demodulando sin ruido
        print("Demod. correcta.")

    noise_signal = awgn(-1, modulated_signal, t) # 3 dB
    out_test = demodulation(f0, f1, t, noise_signal, bit_rate, f_s)

    if out_test == bit_signal: # Se prueba demodulando con ruido
        print("Demod. correcta.")

    plot_time(t, noise_signal, "Señal Ruidosa (largo pequeño)", "ruidomod10")

    # Simulacion (4)

    #print(check_errors([0, 1, 1, 1], [1, 1, 1, 1]))

    # bit_rates = [300] # [100, 200, 300]
    # # db_ranges = [i for i in range(-2, 9) if i != 0]  se excluye el 0 y 9
    # db_ranges = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5]

    # random_bits = get_random_bits(10**5)

    # errors = []
    # outer_t0 = time.time()
    # for bit_rate in bit_rates:
    #     errs = []
    #     for db in db_ranges:
    #         f0, f1, t, modulated_signal = modulation(random_bits, bit_rate, f_s)
    #         noise_signal = awgn(db, modulated_signal, t)

    #         out_test = demodulation(f0, f1, t, random_bits, noise_signal, bit_rate, f_s)

    #         ber = check_errors(random_bits, out_test)
    #         errs.append(ber)

    #     errors.append(errs)

    # outer_t1 = time.time()
    # print("t = ", outer_t1 - outer_t0)
    # for err in errors:
    #     print(err)

