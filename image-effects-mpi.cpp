#include <opencv4/opencv.hpp>
#include <opencv4/imgcodecs.hpp>
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

void grisPromedioOMP(Mat Ptr_src,Mat Ptr_dst, int threads){

    uint8_t* pixelPtr_src = (uint8_t*)Ptr_src.data; //Puntero imagen original
    uint8_t* pixelPtr_dst = (uint8_t*)Ptr_dst.data; //Puntero imagen destino
    int cn = Ptr_src.channels();//Numero de canales

    int width = Ptr_src.rows;
    int height = Ptr_dst.cols;
    omp_set_dynamic(0);
    #pragma omp parallel for num_threads(threads) collapse(2)
	for(int i = 0; i <= width; i++){
	    
        for(int j = 0; j < height; j++){
            uint8_t pixel[cn];
            //obtenemos valores RGB de la imagen
            pixel[0] = (uint8_t)pixelPtr_src[i*height*cn + j*cn + 0]; // B
            pixel[1] = (uint8_t)pixelPtr_src[i*height*cn + j*cn + 1]; // G
            pixel[2] = (uint8_t)pixelPtr_src[i*height*cn + j*cn + 2]; // R
            //hacemos la logica del filtro con los valores RGB
            uint8_t Grey = (pixel[0]+pixel[1]+pixel[2])/3;
            //asignamos el valor calculado al unico canal de la imagen a crear
            pixelPtr_dst[i * height + j] = Grey;
        }
    } 
}

void grisLumaOMP(Mat Ptr_src,Mat Ptr_dst,int threads){
    int width = Ptr_src.rows;
    int height = Ptr_dst.cols;
    
    uint8_t* pixelPtr_src = (uint8_t*)Ptr_src.data; //Puntero imagen original
    uint8_t* pixelPtr_dst = (uint8_t*)Ptr_dst.data; //Puntero imagen destino
    int cn = Ptr_src.channels();//Numero de canales
	
	
    omp_set_dynamic(0);
    #pragma omp parallel for num_threads(threads) collapse(2)
	for(int i = 0; i <= width; i++){
        for(int j = 0; j < height; j++){
            uint8_t pixel[cn]; 
            //obtenemos valores RGB de la imagen
            pixel[0] = (uint8_t)pixelPtr_src[i*height*cn + j*cn + 0]; // B
            pixel[1] = (uint8_t)pixelPtr_src[i*height*cn + j*cn + 1]; // G
            pixel[2] = (uint8_t)pixelPtr_src[i*height*cn + j*cn + 2]; // R
            //hacemos la logica del filtro con los valores RGB
            uint8_t Grey = (pixel[0]*0.0722+pixel[1]*0.7152+pixel[2]*0.2126);
            //asignamos el valor calculado al unico canal de la imagen a crear
            pixelPtr_dst[i * height + j] = Grey;
        }
    }
}

void sombrasDeGrisOMP(Mat Ptr_src,Mat Ptr_dst,int capas,int threads){
    
    if(capas<2) capas = 2;
    if(capas>255) capas = 255;
    int ConversionFactor = 255 / (capas - 1);
    
    int width = Ptr_src.rows;
    int height = Ptr_dst.cols;
    
    uint8_t* pixelPtr_src = (uint8_t*)Ptr_src.data; //Puntero imagen original
    uint8_t* pixelPtr_dst = (uint8_t*)Ptr_dst.data; //Puntero imagen destino
    int cn = Ptr_src.channels();//Numero de canales
	 
	
    omp_set_dynamic(0);
    #pragma omp parallel for num_threads(threads) collapse(2)
	for(int i = 0; i <= width; i++){
        for(int j = 0; j < height; j++){
            uint8_t pixel[cn];
            //obtenemos valores RGB de la imagen
            pixel[0] = (uint8_t)pixelPtr_src[i*height*cn + j*cn + 0]; // B
            pixel[1] = (uint8_t)pixelPtr_src[i*height*cn + j*cn + 1]; // G
            pixel[2] = (uint8_t)pixelPtr_src[i*height*cn + j*cn + 2]; // R

            //hacemos la logica del filtro con los valores RGB
            uint8_t AverageValue = (pixel[0]+pixel[1]+pixel[2])/3;
            
            uint8_t Gray = (int)((AverageValue / ConversionFactor) + 0.5) * ConversionFactor;
            //asignamos el valor calculado al unico canal de la imagen a crear
            pixelPtr_dst[i * height + j] = Gray;
        }
    }
}

void granularOMP(Mat Ptr_src,Mat Ptr_dst,int capas, int threads){
    if(capas<2) capas = 2;
    if(capas>255) capas = 255;

    int ConversionFactor = 255 / (capas - 1);
    
    int width = Ptr_src.rows;
    int height = Ptr_dst.cols;


    
    uint8_t* pixelPtr_src = (uint8_t*)Ptr_src.data; //Puntero imagen original
    uint8_t* pixelPtr_dst = (uint8_t*)Ptr_dst.data; //Puntero imagen destino
    int cn = Ptr_src.channels();//Numero de canales
	 
	

    omp_set_dynamic(0);
    #pragma omp parallel for num_threads(threads)
	for(int i = 0; i <= width; i++){
        long errorValue = 0;
        for(int j = 0; j < height; j++){
            uint8_t pixel[cn];
            //obtenemos valores RGB de la imagen
            pixel[0] = (uint8_t)pixelPtr_src[i*height*cn + j*cn + 0]; // B
            pixel[1] = (uint8_t)pixelPtr_src[i*height*cn + j*cn + 1]; // G
            pixel[2] = (uint8_t)pixelPtr_src[i*height*cn + j*cn + 2]; // R

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
            pixelPtr_dst[i * height + j] = grey;
        }
        errorValue = 0;
    }
}


int main(int argc, char *argv[]) {
    //declaramos argumentos de entradas
    char* nombre_src;
    nombre_src = (char*)malloc(sizeof(char)*20);
    char* nombre_dst;
    nombre_dst = (char*)malloc(sizeof(char)*20);
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




    //declaramos variable para tomar el tiempo
	struct timeval* tval_before,* tval_after,* tval_result;
    tval_before = (struct timeval*)malloc(sizeof(struct timeval));
    tval_after = (struct timeval*)malloc(sizeof(struct timeval));
    tval_result = (struct timeval*)malloc(sizeof(struct timeval));

    //Creamos las variables de pthread de acuerdo a los arg
    pthread_t threads[hilos];
    struct filter_data td[hilos];
    
    //Leemos la imagen
    Mat imagen_src = imread(nombre_src);
    free(nombre_src);
    //Advertimos si no se encuentra la imagen
    if( imagen_src.empty() ){
        printf(" Error opening image\n");
        return -1;
    }

    //Declaramos objeto sobre el cual se trabajara la imagen a crear
    Mat image_dst(imagen_src.rows, imagen_src.cols,CV_8UC1, Scalar(0));

    int height = imagen_src.rows;
    int width = imagen_src.cols;
    int rc;

    switch (parametro_filtro){
        //filtro grisPromedio
        case 1:
            gettimeofday(tval_before, NULL);
            for(int i = 0; i < hilos; i++){
                //segun la cantidad de hilos dividimos las iteraciones por block-wise y se las pasamos a una estructura para pasar la info al hilo
                td[i].inicio = (int)(height/hilos)*i;
                td[i].fin = ((int)(height/hilos)*(i+1))-1;
                if (i==(hilos-1)) {
                    td[i].fin = height-1;
                }
                td[i].Ptr_src = imagen_src;
                td[i].Ptr_dst = image_dst;
                td[i].capas = 0;
                rc = pthread_create(&threads[i], NULL, grisPromedio, (void *)&td[i]);
                if (rc) {
                    cout << "Error:unable to create thread," << rc << endl;
                    exit(-1);
                }
            }
            for(int i = 0; i < hilos; i++){
                pthread_join( threads[i], NULL);
            }
            gettimeofday(tval_after, NULL);
            timersub(tval_after, tval_before, tval_result);
            printf("POSIX: %ld.%06ld\n", (long int)tval_result->tv_sec, (long int)tval_result->tv_usec);
            
            //OMP
            
            gettimeofday(tval_before, NULL);
            grisPromedioOMP(imagen_src,image_dst,hilos);
            gettimeofday(tval_after, NULL);
            timersub(tval_after, tval_before, tval_result);
            printf("OMP: %ld.%06ld\n", (long int)tval_result->tv_sec, (long int)tval_result->tv_usec);
            
            
            
            
            break;
        //filtro Luma
        case 2:
            gettimeofday(tval_before, NULL);
            for(int i = 0; i < hilos; i++){
                //segun la cantidad de hilos dividimos las iteraciones por block-wise y se las pasamos a una estructura para pasar la info al hilo
                td[i].inicio = (int)(height/hilos)*i;
                td[i].fin = ((int)(height/hilos)*(i+1))-1;
                if (i==(hilos-1)) {
                    td[i].fin = height-1;
                }
                td[i].Ptr_src = imagen_src;
                td[i].Ptr_dst = image_dst;
                td[i].capas = 0;
                rc = pthread_create(&threads[i], NULL, grisLuma, (void *)&td[i]);
                if (rc) {
                    cout << "Error:unable to create thread," << rc << endl;
                    exit(-1);
                }
            }
            for(int i = 0; i < hilos; i++){
                pthread_join( threads[i], NULL);
            }
            gettimeofday(tval_after, NULL);
            timersub(tval_after, tval_before, tval_result);
            printf("POSIX: %ld.%06ld\n", (long int)tval_result->tv_sec, (long int)tval_result->tv_usec);
            //OMP
            
            gettimeofday(tval_before, NULL);
            grisLumaOMP(imagen_src,image_dst,hilos);
            gettimeofday(tval_after, NULL);
            timersub(tval_after, tval_before, tval_result);
            printf("OMP: %ld.%06ld\n", (long int)tval_result->tv_sec, (long int)tval_result->tv_usec);
            
            break;
        //filtro sombrasDeGris
        case 3:
        
            gettimeofday(tval_before, NULL);
            for(int i = 0; i < hilos; i++){
                //segun la cantidad de hilos dividimos las iteraciones por block-wise y se las pasamos a una estructura para pasar la info al hilo
                td[i].inicio = (int)(height/hilos)*i;
                td[i].fin = ((int)(height/hilos)*(i+1))-1;
                if (i==(hilos-1)) {
                    td[i].fin = height-1;
                }
                td[i].Ptr_src = imagen_src;
                td[i].Ptr_dst = image_dst;
                td[i].capas = capas;
                rc = pthread_create(&threads[i], NULL, sombrasDeGris, (void *)&td[i]);
                if (rc) {
                    cout << "Error:unable to create thread," << rc << endl;
                    exit(-1);
                }
            }
            for(int i = 0; i < hilos; i++){
                pthread_join( threads[i], NULL);
            }
            gettimeofday(tval_after, NULL);
            timersub(tval_after, tval_before, tval_result);
            printf("POSIX: %ld.%06ld\n", (long int)tval_result->tv_sec, (long int)tval_result->tv_usec);
            //OMP
            
            gettimeofday(tval_before, NULL);
            sombrasDeGrisOMP(imagen_src,image_dst,capas,hilos);
            gettimeofday(tval_after, NULL);
            timersub(tval_after, tval_before, tval_result);
            printf("OMP: %ld.%06ld\n", (long int)tval_result->tv_sec, (long int)tval_result->tv_usec);
            
            break;
        //filtro granular
        case 4:
            gettimeofday(tval_before, NULL);
            for(int i = 0; i < hilos; i++){
                //segun la cantidad de hilos dividimos las iteraciones por block-wise y se las pasamos a una estructura para pasar la info al hilo
                td[i].inicio = (int)(height/hilos)*i;
                td[i].fin = ((int)(height/hilos)*(i+1))-1;
                if (i==(hilos-1)) {
                    td[i].fin = height-1;
                }
                td[i].Ptr_src = imagen_src;
                td[i].Ptr_dst = image_dst;
                td[i].capas = capas;
                rc = pthread_create(&threads[i], NULL, granular, (void *)&td[i]);
                if (rc) {
                    cout << "Error:unable to create thread," << rc << endl;
                    exit(-1);
                }
            }
            for(int i = 0; i < hilos; i++){
                pthread_join( threads[i], NULL);
            } 
            gettimeofday(tval_after, NULL);
            timersub(tval_after, tval_before, tval_result);
            printf("POSIX: %ld.%06ld\n", (long int)tval_result->tv_sec, (long int)tval_result->tv_usec);
            
            //OMP
            
            gettimeofday(tval_before, NULL);
            granularOMP(imagen_src,image_dst,capas,hilos);
            gettimeofday(tval_after, NULL);
            timersub(tval_after, tval_before, tval_result);
            printf("OMP: %ld.%06ld\n", (long int)tval_result->tv_sec, (long int)tval_result->tv_usec);
            
            break;
    
        default:
            break;
    }
    //Guardamos la imagen
    imwrite(nombre_dst,image_dst);
    free(nombre_dst);
    return 1;
}
