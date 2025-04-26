from PIL import Image  # Para manejar imágenes
import numpy as np  # Para operaciones con arreglos
import time  # Para medir el tiempo de ejecución

# Genera un kernel gaussiano dado un tamaño y sigma
def generar_kernel_gaussiano(tamaño, sigma):
    offset = tamaño // 2  
    kernel = np.zeros((tamaño, tamaño), dtype=np.float32)  
    for y in range(-offset, offset + 1):
        for x in range(-offset, offset + 1):
            # Calcula el valor gaussiano para cada posición
            valor = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
            kernel[y + offset, x + offset] = valor
    # Normaliza el kernel
    kernel /= np.sum(kernel)  
    return kernel

# Aplica el filtro gaussiano de forma secuencial a una imagen en escala de grises
def aplicar_filtro_gaussiano_secuencial(gray_array, kernel):
    # Dimensiones de la imagen
    altura, ancho = gray_array.shape  
    offset = kernel.shape[0] // 2  
    # Imagen de salida
    resultado = np.zeros_like(gray_array, dtype=np.uint8)  

    for y in range(altura):
        for x in range(ancho):
            suma = 0.0  
            for ky in range(-offset, offset + 1):
                for kx in range(-offset, offset + 1):
                    # Coordenadas del píxel con control de bordes
                    pixelX = min(max(x + kx, 0), ancho - 1)
                    pixelY = min(max(y + ky, 0), altura - 1)
                    # Suma ponderada del píxel por el kernel
                    suma += gray_array[pixelY, pixelX] * kernel[ky + offset, kx + offset]
            resultado[y, x] = min(max(int(suma), 0), 255)  # Asigna el valor resultante
    return resultado

# Función principal
def main():
    inicio = time.time()  # Marca el inicio

    # Rutas de entrada y salida
    input_path = "C:/Users/andre/Documents/COMPUTACION_PARALELA/PYTHON/Tarea_Filtro_Python/img.jpg"
    output_path = "C:/Users/andre/Documents/COMPUTACION_PARALELA/PYTHON/Tarea_Filtro_Python/img9.jpg"

    # Abre la imagen
    imagen = Image.open(input_path)  
    gray_array = np.array(imagen)  

    kernel_size = 9  # Tamaño del kernel
    sigma = 1.5  # Valor de sigma para la distribución gaussiana
    kernel = generar_kernel_gaussiano(kernel_size, sigma)  # Genera el kernel
   
   # Aplica el filtro
    resultado_array = aplicar_filtro_gaussiano_secuencial(gray_array, kernel)  
    
    # Crea imagen final
    resultado_img = Image.fromarray(resultado_array.astype(np.uint8), mode='L')  
    resultado_img.save(output_path) 
    fin = time.time()

    print("Filtro gaussiano aplicado con kernel de 13*13")  
    print(f"Tiempo de ejecución: {int((fin - inicio) * 1000)} ms")  

if __name__ == "__main__":
    main()
