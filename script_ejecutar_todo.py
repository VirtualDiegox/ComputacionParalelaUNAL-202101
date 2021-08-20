#!/usr/bin/env python3

import subprocess
import csv


def Average(lst):
    return sum(lst) / len(lst)

imagenes = ["720p", "1080p", "4K"]
extension = ".jpg"
filtros = [1,2,3,4]
num_hilos = [1,2,4,8,16]
capas = 3
data = {}
times_posix = []
times_omp = []
intentos_posix = []
intentos_omp = []

csv_posix = open("datacsv/data_posix.csv", "w")
csv_omp = open("datacsv/data_omp.csv", "w")

writer_posix = csv.writer(csv_posix)
writer_omp = csv.writer(csv_omp)

for imagen in imagenes:
    for hilos in num_hilos:
        for filtro in filtros:
            if filtro == 1 or filtro == 2:
                command = "./filtros " + imagen + extension + " " +imagen+"_withfilter"+ str(filtro) +extension + " " +str(filtro) +" "+ str(hilos)
            else:
                command = "./filtros " + imagen + extension + " " +imagen+"_withfilter"+ str(filtro) +extension + " " + str(filtro)+" "+ str(capas) +" "+ str(hilos)

            for n in range(0,5):
                output = subprocess.Popen(command,shell=True,stdout=subprocess.PIPE)
                print(command)
                string = output.stdout.read().decode('utf-8')
                parts = string.split()
                if len(parts) == 1:
                    time_posix = parts[0]
                    time_omp = parts[0]
                else:
                    time_posix = parts[1]
                    time_omp = parts[3]
                intentos_posix.append(float(time_posix))
                intentos_omp.append(float(time_omp))
        
            times_posix.append(Average(intentos_posix))
            times_omp.append(Average(intentos_omp))
            intentos_posix = []
            intentos_omp = []
        writer_posix.writerow(times_posix)
        writer_omp.writerow(times_omp)

        times_omp = []
        times_posix = []


                
                

