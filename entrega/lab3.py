from math import floor
import numpy as np
import random as rd
import matplotlib.pyplot  as plt
from scipy import signal, fftpack
import time

def get_fourier_comps(signal, f_s):
    """Función que calcula los componentes de frecuencia de una señal en el tiempo
       y también sus valores de y correspondientes
    Args:
        signal (array): señal en el tiempo
        f_s (int): frecuencia de muestreo

    Returns:
        freqs, fft: el eje x (frecuencias) y eje y (amplitud) de la fft
    """
    fft = fftpack.fft(signal)
    freqs = fftpack.fftfreq(len(fft), d = 1/f_s)
    return freqs, fft

def get_random_bits(n):
    """Genera un arreglo aleatorio de bits, de largo n.
    Args:
        n (int): largo requerido del arreglo de bits.
    Returns:
        array: arreglo de bits aleatorio.
    """
    return [rd.randint(0, 1) for i in range(n)]

def double_plot(t, c0, c1, title, filename, noise = False):
    """Gráfica del estado de las portadoras en algún momento de mod/demod.

    Args:
        t (array): vector de tiempo
        c0 (array): portadora 1
        c1 (array)): portadora 2
        title (str): titulo del plot
        filename (str): nombre del archivo a guardar sin extensión
        noise (bool, optional): determina si el gráfico contiene el ruido.
    """
    f = plt.figure()
    f.suptitle(title)
    f.text(0.5, 0.04, "t (segundos)", ha = 'center')
    f.text(0.04, 0.5, "Amplitud", va = 'center', rotation = 'vertical')

    plt.subplot(2, 1, 1)
    plt.plot(t, c0)
    plt.subplot(2, 1, 2)
    plt.plot(t, c1)
    # plt.show()

    if noise:
        filename = filename + "_noise"
    plt.savefig(filename+".png")

    plt.clf()

    return None

def fft_plot(signal, f_s, title, filename):
    """Grafica el espectro de frecuencias a través de la FFT de una señal.

    Args:
        signal (array): señal en el tiempo
        f_s (num): freq. de muestreo
        title (str): titulo del plot
        filename (str): nombre del plot
    """
    freqs, ft = get_fourier_comps(signal, f_s)
    plt.plot(fftpack.fftshift(freqs), fftpack.fftshift(np.abs(ft)))
    plt.title(title)
    plt.xlabel("Freq. (Hz)")
    plt.ylabel("Amplitud")
    plt.savefig(filename + '.png')
    plt.clf()
    return None

def plot_time(t, y, title, filename):
    """Grafica de una señal en el tiempo.

    Args:
        t (array): vector tiempo
        y (array): señal
        title (str): titulo del plot
        filename (str): nombre del archivo a guardar, sin extensión
    """
    plt.plot(t, y)
    plt.ylabel("Amplitud")
    plt.xlabel("tiempo (segundos)")
    plt.title(title)
    # plt.show()
    plt.savefig(filename + ".png")
    plt.clf()

def demod_plots(t, modulated_signal, s0, s1, title, filename, noise = False):
    """Función que grafica los estados intermedios de la demodulación, como
       cuando se está correlacionando o sacando la envolvente.

    Args:
        t (array): vector temporal total
        modulated_signal (array): señal modulada
        s0 (array): señal en demod. intermedia para graficar
        s1 (array): señal en demod. intermedia para graficar
        title (str): titulo del plot
        filename (str): nombre del plot sin extensión
        noise (bool, opt): determina si se grafica con ruido o no.

    """
    f = plt.figure()
    f.suptitle(title)
    f.text(0.5, 0.04, "t (segundos)", ha = 'center')
    f.text(0.04, 0.5, "Amplitud", va = 'center', rotation = 'vertical')

    plt.subplot(3, 1, 1)
    plt.plot(t, modulated_signal)

    plt.subplot(3, 1, 2)
    plt.plot(t, s0)

    plt.subplot(3, 1, 3)
    plt.plot(t, s1)

    if noise:
        filename = filename + "_noise"
    plt.savefig(filename+".png")

    return None

def modulation(bit_signal, bit_rate, f_s, simul = False):
    """Función que toma un arreglo de bits y los modula a cierto bit rate 
       y a una freq. de muestreo dada.
    Args:
        bit_signal (array): secuencia binaria (0, 1)
        bit_rate (num): tasa de datos de transmisión
        f_s (num): frecuencia de muestreo para las portadoras >> max(f0, f1)
        simul (bool, optional): se realiza dentro de simulación o no.

    Returns:
        f0: frecuencia portadora para 0
        f1: frecuencia portadora para 1
        t: vector de tiempo para tiempo de transmisión total
        y: señal modulada, esquema FSK
    """
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

    if not simul:
        fft_plot(y, f_s, f"FFT señal modulada, peaks en {f0} y {f1} (Hz)", "fftmod")
        double_plot(t_c, carrier0, carrier1, 
                    f"Señales Portadoras en el tiempo, a frecuencias {f0} y {f1} (Hz)",
                    "portadoras")

    return f0, f1, total_time, y

def demodulation(f0, f1, t, modulated_signal, bit_rate, f_s, simul = False, noise = False):
    """Función que demodula una señal de entrada FSK.
    Args:
        f0 (int): freq. de portadora para símbolo 0
        f1 (int): freq. portadora símbolo 1
        t (array): vector temporal, en segundos
        modulated_signal (array): señal modulada con FSK
        bit_rate (int): bit rate en bps
        f_s (int): sampling rate de señales, >> a frecuencias portadoras
        simul (bool): se realiza dentro de simulación o no.
        noise (bool): señal es ruidosa o no.
    Returns:
        ---: ---
    """
    bit_time = 1/bit_rate
    endpoint = 1/f_s
    t_c = np.arange(0, bit_time + endpoint, endpoint)
    c0 = np.cos(2*np.pi*f0*t_c)
    c1 = np.cos(2*np.pi*f1*t_c)

    m = 'same' # parámetro por defecto solo devuelve un escalar
    s0 = np.correlate(modulated_signal, c0, mode = m)
    s1 = np.correlate(modulated_signal, c1, mode = m)

    if not simul:
        demod_plots(t, modulated_signal, s0, s1, 
                        "Correlación en ambas señales (tiempo)", "correl", noise)
        

    s0 = signal.hilbert(np.gradient(s0))
    s0 = np.abs(s0)

    s1 = signal.hilbert(np.gradient(s1))
    s1 = np.abs(s1)

    if not simul:
        double_plot(t, s0, s1, 
                   "Detección de envolvente (tiempo)", "envelop", noise)
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
    """Generador de AWGN para una señal.
    Args:
        SNR_rate (num): SNR en dB
        modulated_signal (array): señal modulada, en FSK
        t (array): vector temporal

    Returns:
        array: señal ruidosa
    """
    p_signal = sum(np.abs(modulated_signal)**2)/(2*len(modulated_signal))
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
    """Calcula cuántos bits difieren entre secuencias binarias originales y la demodulada con rudio.
       Se utilizan arreglos de numpy para acelerar esta parte.
    Args:
        original_bits (array)
        demod_bits (array)

    Returns:
        num: ratio # bits distintos / # total de bits
    """
    count = np.sum(np.array(original_bits) != np.array(demod_bits))
    return count/len(original_bits)


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

    noise_signal = awgn(-2, modulated_signal, t) # -2 dB
    out_test = demodulation(f0, f1, t, noise_signal, bit_rate, f_s, noise = True)

    if out_test == bit_signal: # Se prueba demodulando con ruido
        print("Demod. correcta.")

    plot_time(t, noise_signal, "Señal Ruidosa (largo pequeño)", "ruidomod10")

    # Simulacion (4)

    bit_rates = [300, 600, 900] 
    db_ranges = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    random_bits = get_random_bits(10**5)
    errors = []
    outer_t0 = time.time()
    for bit_rate in bit_rates:
        errs = []
        for db in db_ranges:
            f0, f1, t, modulated_signal = modulation(random_bits, bit_rate, f_s, simul = True)
            noise_signal = awgn(db, modulated_signal, t)

            out_test = demodulation(f0, f1, t, noise_signal, bit_rate, f_s, simul = True)

            ber = check_errors(random_bits, out_test)
            errs.append(ber)
        errors.append(errs)
    outer_t1 = time.time()

    print("t = ", outer_t1 - outer_t0)

    # Graficamos resultados de simulación
    
    for err in errors:
        plt.semilogy(db_ranges, err)
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title("Simul. con arreglo de 10E5")
    plt.savefig("simulacion.png")
    plt.clf()
