#!/usr/bin/env python3

import subprocess
import csv


def Average(lst):
    return sum(lst) / len(lst)

imagenes = ["720p","1080p","4K"]
extension = ".jpg"
filtros = [1,2,3]
num_hilos = [1,2,4,8,16,32,64,128,256]
num_bloques = [1,2,4,8,16,32]
capas = 3
data = {}
times_cuda = []
intentos_cuda = []


csv_cuda = open("data_cuda.csv", "w")

writer_cuda = csv.writer(csv_cuda)


for imagen in imagenes:
    path = "data_cuda_" + imagen + ".csv"
    csv_cuda = open(path,"w")
    writer_cuda = csv.writer(csv_cuda)
    for bloques in num_bloques:
        for hilos in num_hilos:
            for filtro in filtros:
                
                
                if filtro == 1 or filtro == 2:
                    command = "./filtros " + imagen + extension + " " +imagen+"_withfilter"+ str(filtro) +extension + " " +str(filtro) +" "+ str(hilos) +" "+ str(bloques)
                else:
                    command = "./filtros " + imagen + extension + " " +imagen+"_withfilter"+ str(filtro) +extension + " " + str(filtro)+" "+ str(capas) +" "+ str(hilos)+" "+ str(bloques)

                for n in range(0,5):
                    output = subprocess.Popen(command,shell=True,stdout=subprocess.PIPE)
                    print(command)
                    string = output.stdout.read().decode('utf-8')
                    parts = string.split()                  
                    time_cuda = parts[0]
                    intentos_cuda.append(float(time_cuda))
                   
        
                times_cuda.append(Average(intentos_cuda))
            
                intentos_cuda = []
                
        
            writer_cuda.writerow(times_cuda)
            times_cuda = []


                
                

