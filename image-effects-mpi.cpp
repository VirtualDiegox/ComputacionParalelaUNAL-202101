//sudo mpic++ image-effects-mpi.cpp -o filtros_mpi -fopenmp `pkg-config --cflags --libs opencv4`
//mpirun -np 1 --host 10.128.0.3 ./filtros_mpi 4K.jpg 4K_withfilter1.jpg 1 2
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <stdlib.h>
#include <sys/time.h>
#include <stdio.h>
#include <string>
#include <pthread.h>
#include <omp.h>
#include <mpi.h>
 
using namespace std;
using namespace cv;

void grisPromedioOMP(uchar* partialBuffer, int threads,uchar* ansBuffer,int imagePartialSize){

    uint8_t* pixelPtr_src = (uint8_t*)partialBuffer; //Puntero imagen original
    
    omp_set_dynamic(0);
    #pragma omp parallel for num_threads(threads)
	for(int i = 0; i < imagePartialSize; i+=3){
	    
            uint8_t pixel[3];
            //obtenemos valores RGB de la imagen
            pixel[0] = (uint8_t)pixelPtr_src[i]; // B
            pixel[1] = (uint8_t)pixelPtr_src[i+1]; // G
            pixel[2] = (uint8_t)pixelPtr_src[i+2]; // R
            //hacemos la logica del filtro con los valores RGB
            uint8_t Grey = (pixel[0]+pixel[1]+pixel[2])/3;
            //printf("Grey:%d i:%d\n", Grey,i);
            //asignamos el valor calculado al unico canal de la imagen a crear
            ansBuffer[i/3] = (uchar)Grey;
        
    } 
}

void grisLumaOMP(uchar* partialBuffer, int threads,uchar* ansBuffer,int imagePartialSize){
    
    uint8_t* pixelPtr_src = (uint8_t*)partialBuffer; //Puntero imagen original
   
    omp_set_dynamic(0);
    #pragma omp parallel for num_threads(threads)
	for(int i = 0; i <= imagePartialSize; i+=3){
        
            uint8_t pixel[3]; 
            //obtenemos valores RGB de la imagen
            pixel[0] = (uint8_t)pixelPtr_src[i]; // B
            pixel[1] = (uint8_t)pixelPtr_src[i+1]; // G
            pixel[2] = (uint8_t)pixelPtr_src[i+2]; // R
            //hacemos la logica del filtro con los valores RGB
            uint8_t Grey = (pixel[0]*0.0722+pixel[1]*0.7152+pixel[2]*0.2126);
            //asignamos el valor calculado al unico canal de la imagen a crear
            ansBuffer[i/3] = (uchar)Grey;
        
    }
}

void sombrasDeGrisOMP(uchar* partialBuffer, int threads,uchar* ansBuffer,int imagePartialSize,int capas){
    
    if(capas<2) capas = 2;
    if(capas>255) capas = 255;
    int ConversionFactor = 255 / (capas - 1);
    
    
    
    uint8_t* pixelPtr_src = (uint8_t*)partialBuffer; //Puntero imagen original
    
	 
	
    omp_set_dynamic(0);
    #pragma omp parallel for num_threads(threads)
	for(int i = 0; i < imagePartialSize; i+=3){
        
            uint8_t pixel[3];
            //obtenemos valores RGB de la imagen
            pixel[0] = (uint8_t)pixelPtr_src[i]; // B
            pixel[1] = (uint8_t)pixelPtr_src[i+1]; // G
            pixel[2] = (uint8_t)pixelPtr_src[i+2]; // R

            //hacemos la logica del filtro con los valores RGB
            uint8_t AverageValue = (pixel[0]+pixel[1]+pixel[2])/3;
            
            uint8_t Gray = (int)((AverageValue / ConversionFactor) + 0.5) * ConversionFactor;
            //asignamos el valor calculado al unico canal de la imagen a crear
            ansBuffer[i/3] = (uchar)Gray;
        
    }
}

void granularOMP(uchar* partialBuffer, int threads,uchar* ansBuffer,int imagePartialSize,int capas){
    if(capas<2) capas = 2;
    if(capas>255) capas = 255;

    int ConversionFactor = 255 / (capas - 1);
    
    
    uint8_t* pixelPtr_src = (uint8_t*)partialBuffer; //Puntero imagen original


    omp_set_dynamic(0);
    #pragma omp parallel for num_threads(threads)
	for(int i = 0; i < imagePartialSize; i+=3){
        long errorValue = 0;
        
            uint8_t pixel[3];
            //obtenemos valores RGB de la imagen
            pixel[0] = (uint8_t)pixelPtr_src[i]; // B
            pixel[1] = (uint8_t)pixelPtr_src[i+1]; // G
            pixel[2] = (uint8_t)pixelPtr_src[i+2]; // R

            //hacemos la logica del filtro con los valores RGB
            uint8_t grey = (pixel[0]+pixel[1]+pixel[2])/3;
            
            long greyTempCalc = grey;
            greyTempCalc += errorValue;
            greyTempCalc = (int)((greyTempCalc / ConversionFactor) + 0.5) * ConversionFactor;

            errorValue = grey + errorValue - greyTempCalc;
            if(greyTempCalc<0){
                grey = 0;
            }else if (greyTempCalc>255){
                grey = 255;
            }else{
                grey = greyTempCalc;
            }
             
            
            //asignamos el valor calculado al unico canal de la imagen a crear
            ansBuffer[i/3] = (uchar)grey;
        
        errorValue = 0;
    }
}


int main(int argc, char *argv[]) {
    //declaramos argumentos de entradas
    int numprocs, processId;
    
    char* nombre_src;
    nombre_src = (char*)malloc(sizeof(char)*40);
    char* nombre_dst;
    nombre_dst = (char*)malloc(sizeof(char)*40);
    int parametro_filtro,capas,hilos;   
    //tomando argumentos por consola
    strcpy(nombre_src,argv[1]);
    strcpy(nombre_dst,argv[2]);
    parametro_filtro = stoi(argv[3]);
    if(parametro_filtro==3 || parametro_filtro==4){
        capas = stoi(argv[4]);
        hilos = stoi(argv[5]);
    }else{
        hilos = stoi(argv[4]);
    }

    Mat imagen_src;
    //Declaramos objeto sobre el cual se trabajara la imagen a crear
    Mat image_dst;
    size_t imageTotalSize;
    size_t imagePartialSize;
    size_t image_dstPartialSize;
    uchar* partialBuffer;
    uchar* ansBuffer;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &processId);
    //declaramos variable para tomar el tiempo
    struct timeval* tval_before,* tval_after,* tval_result;
    tval_before = (struct timeval*)malloc(sizeof(struct timeval));
    tval_after = (struct timeval*)malloc(sizeof(struct timeval));
    tval_result = (struct timeval*)malloc(sizeof(struct timeval));
    if (processId==0){
        
        //Leemos la imagen
        imagen_src = imread(nombre_src);
        free(nombre_src);
        //Advertimos si no se encuentra la imagen
        if( imagen_src.empty() ){
            printf(" Error opening image\n");
            return -1;
        }
        Mat imagen(imagen_src.rows, imagen_src.cols,CV_8UC1, Scalar(0));
        image_dst = imagen;
        
        imageTotalSize = imagen_src.step[0] * imagen_src.rows;
        
        imagePartialSize = imageTotalSize / numprocs;

        image_dstPartialSize = image_dst.step[0] * image_dst.rows /numprocs; 
        
        
    }

    MPI_Bcast( &imagePartialSize, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD );
    MPI_Bcast( &image_dstPartialSize, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD );
    
    MPI_Barrier( MPI_COMM_WORLD );

    partialBuffer = new uchar[imagePartialSize];
    ansBuffer = new uchar[image_dstPartialSize];

    MPI_Barrier( MPI_COMM_WORLD );

    MPI_Scatter( imagen_src.data, imagePartialSize, MPI_UNSIGNED_CHAR, partialBuffer, imagePartialSize, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD );
    MPI_Barrier( MPI_COMM_WORLD );
    MPI_Scatter( image_dst.data, image_dstPartialSize, MPI_UNSIGNED_CHAR, ansBuffer, image_dstPartialSize, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD );


   

    switch (parametro_filtro){
        //filtro grisPromedio
        case 1:
            //OMP
            if (processId == 0) gettimeofday(tval_before, NULL);
            grisPromedioOMP(partialBuffer,hilos,ansBuffer,imagePartialSize);
            if (processId == 0) gettimeofday(tval_after, NULL);
            if (processId == 0)timersub(tval_after, tval_before, tval_result);
            if (processId == 0) printf("OMP: %ld.%06ld\n", (long int)tval_result->tv_sec, (long int)tval_result->tv_usec);
            MPI_Barrier( MPI_COMM_WORLD );
            break;
        //filtro Luma
        case 2:
            //OMP
            if (processId == 0)gettimeofday(tval_before, NULL);
            grisLumaOMP(partialBuffer,hilos,ansBuffer,imagePartialSize);
            if (processId == 0) gettimeofday(tval_after, NULL);
            if (processId == 0)timersub(tval_after, tval_before, tval_result);
            if (processId == 0) printf("OMP: %ld.%06ld\n", (long int)tval_result->tv_sec, (long int)tval_result->tv_usec);
            MPI_Barrier( MPI_COMM_WORLD );
            break;
        //filtro sombrasDeGris
        case 3:
            //OMP
            if (processId == 0)gettimeofday(tval_before, NULL);
            sombrasDeGrisOMP(partialBuffer,hilos,ansBuffer,imagePartialSize,capas);
            if (processId == 0) gettimeofday(tval_after, NULL);
            if (processId == 0)timersub(tval_after, tval_before, tval_result);
            if (processId == 0) printf("OMP: %ld.%06ld\n", (long int)tval_result->tv_sec, (long int)tval_result->tv_usec);
            MPI_Barrier( MPI_COMM_WORLD );
            break;
        //filtro granular
        case 4:           
            //OMP 
            if (processId == 0)gettimeofday(tval_before, NULL);
            granularOMP(partialBuffer,hilos,ansBuffer,imagePartialSize,capas);
            if (processId == 0) gettimeofday(tval_after, NULL);
            if (processId == 0)timersub(tval_after, tval_before, tval_result);
            if (processId == 0) printf("OMP: %ld.%06ld\n", (long int)tval_result->tv_sec, (long int)tval_result->tv_usec);
            MPI_Barrier( MPI_COMM_WORLD );
            break;
    
        default:
            break;
    }


    MPI_Barrier( MPI_COMM_WORLD );
    MPI_Gather( ansBuffer, image_dstPartialSize, MPI_UNSIGNED_CHAR, image_dst.data, image_dstPartialSize, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD );
    
    if(processId == 0){
        imwrite(nombre_dst,image_dst);
        free(nombre_dst);
        
    }
    
    delete[]partialBuffer;

    MPI_Finalize();
    //Guardamos la imagen
    
    return 0;
}
