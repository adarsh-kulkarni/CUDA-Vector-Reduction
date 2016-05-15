use strict;
use warnings;

my @Blocksize=(32,64,128,256);
my @inputSize=(1000,10000,100000,1000000,2000000);


for (my $i=0; $i < 4; $i++){
	for (my $j=0; $j < 5; $j++){
	
	system "~/NVIDIA_GPU_Computing_SDK/C/bin/linux/release/atomicReduce -size $inputSize[$j] -blocksize $Blocksize[$i] >> atomicReduceResults.txt"
}
}

