############# fill in your assigned years
yr1=2007
yr2=2008
yr3=2009
yr4=2010
yr5=2011
yr6=2012
############

python ./audio_feature_421_${yr1}.py &
python ./audio_feature_421_${yr2}.py &
python ./audio_feature_421_${yr3}.py &
python ./audio_feature_421_${yr4}.py &
python ./audio_feature_421_${yr5}.py &
python ./audio_feature_421_${yr6}.py &
wait
