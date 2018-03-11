/* Spesifikasi:
	Judul/Nama file :GanjilGenap_052.C
	Input (/ IS) :a, integer
	Proses : menuliskan tipe bilangan yang di input
	Output (/ FS):"Tipe Bilangan"
	Dibuat Oleh: Wahyu A P Nainggolan
	Tanggal : 26/02/2015
*/
#include <stdio.h>
int main ()
{
	/* Kamus */
	int a;

	/* Algoritma */
	printf ("Masukkan angka: ");
	scanf ("%d",&a);
	
	if (a % 2 == 0){
		printf ("Bilangan Genap: ");
	}
	else {
		printf ("Bilangan Ganjil: ");
	}
	
return 0;

}
