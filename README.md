#Practicas Computacion Paralela 2021-1

Hecho por dbayonac y gaperezsa

##Practica 1: Filtros B/N con POSIX

Comando para compilar:
```bash
g++ image-effects.cpp -o filtros -lpthread `pkg-config --cflags --libs opencv4`
```
Si tiene una instalacion diferente de opencv basta con cambiar "opencv4" por "opencv" o "opencv2" en los include.

Comando para correr efectos:
```python
./filtros [nombre_src] [nombre_dst] [filtro] [parametros de filtro] [num_hilos]
```
nombre_src: ubicacion y nombre de la imagen fuente
nombre_dst: ubicacion y nombre de la imagen destino
filtro: 1-4 que filtro se quiere aplicar 1-gris promedio 2-luma 3-capas 4-granular
parametros de filtro: si el filtro lo requiere se pasan la cantidad de capas a utilizar, en caso filtro 1 o 2 omitir este parametro
num_hilos:si es 1 se corre secuencialmente, en caso contrario se utilizara la cantidad de hilos especificados

ejemplo: 
```python
./filtros 4K.jpg 4K_withfilter4.jpg 4 3 16
```

Comando para correr script de python:
Permite verificar la funcionalidad completa del programa
Primero se debe compilar el programa c++ como fue especificado para que el script funcione correctamente

```bash
chmod +x script_ejecutar_todo.py
./script_ejecutar_todo.py
```
