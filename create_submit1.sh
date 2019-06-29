#bin/bash
name="youku_00200_00250_"
i=200

for file in $(ls "./Results")
do
echo ${file}
if [[ "${i}" -lt "205" ]]
then
	ffmpeg -i ./Results/${name}${i}/sr_%3d.bmp -pix_fmt yuv420p -vsync 0 ./submit/Youku_00${i}_h_Res.y4m
else
	cp ./Results/${name}${i}/sr_001.bmp ./videos/sr_001.bmp
	cp ./Results/${name}${i}/sr_026.bmp ./videos/sr_002.bmp
	cp ./Results/${name}${i}/sr_051.bmp ./videos/sr_003.bmp
	cp ./Results/${name}${i}/sr_076.bmp ./videos/sr_004.bmp	
	ffmpeg -i ./videos/sr_%3d.bmp -pix_fmt yuv420p -vsync 0 ./submit/Youku_00${i}_h_Sub25_Res.y4m

fi
i=$((1 + ${i}))
done
cd submit
zip result.zip *.y4m
	
