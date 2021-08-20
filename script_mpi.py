#!/usr/bin/env python3

import subprocess
import csv

def Average(lst):
    return sum(lst) / len(lst)

imagenes = ["720p", "1080p", "4K"]
extension = ".jpg"
filtros = [1,2,3,4]
num_hilos = [1,2,4,8,16]
num_nodes = [1,2,4,8]
capas = 3
times_omp = []
intentos_omp = []
hostfile = "mpi_hosts"

csv_omp = open("datacsv/data_omp.csv", "w")
writer_omp = csv.writer(csv_omp)

for nodes in num_nodes:
    for imagen in imagenes:
        for hilos in num_hilos:
            for filtro in filtros:
