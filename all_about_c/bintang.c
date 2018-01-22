#include<stdio.h>
int main(){
	

int row,bintang,spasi;
int n;
 printf("masukkan banyaknya nilai n =");
 scanf("%d",&n);
 int banyakBintang = n;
 for(row=1;row<=banyakBintang;row++){
		 	for (spasi=1;spasi<=n;spasi++){
		 		printf(" ");
			 }
			 for (bintang=1;bintang<=row;bintang++){
			 	 printf(" ");
				 printf("*");
			 }
			 printf("\n");
			 n=n-1;
 		}
 return 0;
 }
 
 
