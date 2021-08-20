# Practicas Computacion Paralela 2021-1

Hecho por:
* Diego Alejandro Bayona - dbayonac 
* Gabriel Perez Santamaria - gaperezsa

## Filtros B/N con POSIX, OMP y secuenciales

Se implementan 4 filtros de escala de grises para 3 tamanos de imagen

### Librerias Necesarias

* OpenCV
* lpthread

### Ejecucion

Comando para compilar:
```bash
nvcc image-effects.cu -o filtros `pkg-config --cflags --libs opencv4`
```
Si tiene una instalacion diferente de opencv basta con cambiar "opencv4" por "opencv" o "opencv2" en los include.

Comando para correr efectos:
```python
./filtros [nombre_src] [nombre_dst] [filtro] [parametros de filtro] [num_hilos]
```
* nombre_src: ubicacion y nombre de la imagen fuente
* nombre_dst: ubicacion y nombre de la imagen destino
* filtro: 1-4 que filtro se quiere aplicar:
    * 1 gris promedio
    * 2 luma 
    * 3 capas 
    * 4 granular
* parametros de filtro: si el filtro lo requiere se pasan la cantidad de capas a utilizar, en caso filtro 1 o 2 omitir este parametro
* num_hilos: si es 1 se corre secuencialmente, en caso contrario se utilizara la cantidad de hilos especificados

Ejemplo: 
```python
./filtros 4K.jpg 4K_withfilter4.jpg 4 3 16
```

Comando para correr script de python:
Permite verificar la funcionalidad completa del programa\n
y guarda los datos en un csv indicado
Primero se debe compilar el programa c++ como fue especificado para que el script funcione correctamente

```bash
chmod +x script_ejecutar_todo.py
./script_ejecutar_todo.py
```

## Filtros B/N con CUDA 

Se implementan 3 filtros de escala de grises para 3 tamanos de imagen utilizando CUDA

### Librerias Necesarias

* CUDA
* OpenCV junto a las librerias extra para trabajar con CUDA

### Ejecucion

Comando para compilar:
```bash
nvcc image-effects.cu -o filtros `pkg-config --cflags --libs opencv4`
```
Comando para correr efectos:
```python
./filtros [nombre_src] [nombre_dst] [filtro] [parametros de filtro] [num_hilos] [num_bloques]
```
* nombre_src: ubicacion y nombre de la imagen fuente
* nombre_dst: ubicacion y nombre de la imagen destino
* filtro: 1-4 que filtro se quiere aplicar:
    * 1 gris promedio
    * 2 luma 
    * 3 capas 
    * 4 granular
* parametros de filtro: si el filtro lo requiere se pasan la cantidad de capas a utilizar, en caso filtro 1 o 2 omitir este parametro
* num_hilos: se utilizara la cantidad de hilos especificados
* num_bloques: se utilizara la cantidad de bloques especificados


Comando para correr script de python:
Permite verificar la funcionalidad completa del programa\n
y guarda los datos en un csv indicado
Primero se debe compilar el programa nvcc como fue especificado para que el script funcione correctamente

```bash
chmod +x scrip_cuda.py
./scrip_cuda.py
```