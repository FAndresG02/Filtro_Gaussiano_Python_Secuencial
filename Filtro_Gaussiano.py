from PIL import Image
import numpy as np
import time

def generar_kernel_gaussiano(tamaño, sigma):
    offset = tamaño // 2
    kernel = np.zeros((tamaño, tamaño), dtype=np.float32)
    for y in range(-offset, offset + 1):
        for x in range(-offset, offset + 1):
            valor = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
            kernel[y + offset, x + offset] = valor
    kernel /= np.sum(kernel)
    return kernel

def aplicar_filtro_gaussiano_secuencial(gray_array, kernel):
    altura, ancho = gray_array.shape
    offset = kernel.shape[0] // 2
    resultado = np.zeros_like(gray_array, dtype=np.uint8)

    for y in range(altura):
        for x in range(ancho):
            suma = 0.0
            for ky in range(-offset, offset + 1):
                for kx in range(-offset, offset + 1):
                    pixelX = min(max(x + kx, 0), ancho - 1)
                    pixelY = min(max(y + ky, 0), altura - 1)
                    suma += gray_array[pixelY, pixelX] * kernel[ky + offset, kx + offset]
            resultado[y, x] = min(max(int(suma), 0), 255)

    return resultado

def main():
    inicio = time.time()

    input_path = "C:/Users/andre/Documents/COMPUTACION_PARALELA/PYTHON/Tarea_Filtro_Python/img.jpg"
    output_path = "C:/Users/andre/Documents/COMPUTACION_PARALELA/PYTHON/Tarea_Filtro_Python/img9.jpg"

    imagen = Image.open(input_path)  
    gray_array = np.array(imagen)

    kernel_size = 9
    sigma = 1.5
    kernel = generar_kernel_gaussiano(kernel_size, sigma)

    resultado_array = aplicar_filtro_gaussiano_secuencial(gray_array, kernel)

    resultado_img = Image.fromarray(resultado_array.astype(np.uint8), mode='L')
    resultado_img.save(output_path)

    fin = time.time()

    print("Filtro gaussiano aplicado con kernel de 13*13")
    print(f"Tiempo de ejecución: {int((fin - inicio) * 1000)} ms")

if __name__ == "__main__":
    main()
