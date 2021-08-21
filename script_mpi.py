#!/usr/bin/env python3

import subprocess
import csv

def Average(lst):
    return sum(lst) / len(lst)

imagenes = ["720p", "1080p", "4K"]
extension = ".jpg"
filtros = [1,2,3]
num_hilos = [1,2,4,8,16]
num_nodes = [1,2,4,8]
capas = 3
times_mpi = []
intentos_mpi = []
hostfile = "mpi_hosts"

csv_mpi = open("datacsv/data_mpi.csv", "w")
writer_mpi = csv.writer(csv_mpi)

for nodes in num_nodes:
    for imagen in imagenes:
        for hilos in num_hilos:
            for filtro in filtros:
                if filtro == 1 or filtro == 2:
                    command = "mpirun -np "+ str(nodes) + " --hostfile " + hostfile + " ./filtros_mpi " + "images/" + imagen + extension + " " + "images_withfilter/" + imagen + "_withfilter"+ str(filtro) + extension + " " +str(filtro) +" "+ str(hilos)
                else:
                    command = "mpirun -np "+ str(nodes) + " --hostfile " + hostfile + " ./filtros_mpi " + "images/" + imagen + extension + " " + "images_withfilter/" + imagen + "_withfilter"+ str(filtro) + extension + " " +str(filtro) +" "+ str(capas) + " " + str(hilos)

                for n in range(0,5):
                    output = subprocess.Popen(command,shell=True,stdout=subprocess.PIPE)
                    print(command)
                    string = output.stdout.read().decode('utf-8')
                    parts = string.split()                  
                    time_mpi = parts[1]
                    intentos_mpi.append(float(time_mpi))
                times_mpi.append(Average(intentos_mpi))
                intentos_mpi = []
            writer_mpi.writerow(times_mpi)
            times_mpi = []
