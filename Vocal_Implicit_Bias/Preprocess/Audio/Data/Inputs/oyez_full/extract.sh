base="fullTierName"
ext=".csv"

grep "Year" "$base$ext" > "${base}_1998$ext"
grep 1998 "$base$ext" >> "${base}_1998$ext"

grep "Year" "$base$ext" > "${base}_1999$ext"
grep 1999 "$base$ext" >> "${base}_1999$ext"

for yr in 03 04 05 06 07 08 09 10 11 12
do
 echo "greping $yr"
 grep "Year" "$base$ext" > "${base}_20${yr}$ext"
 grep "20$yr" "$base$ext" >> "${base}_20${yr}$ext"
done
