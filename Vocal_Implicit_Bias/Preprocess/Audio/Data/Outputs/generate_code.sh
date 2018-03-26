base="audio_feature"
ext=".py"

cp "$base$ext" "${base}_1998$ext"

sed -i '.bak' 's/fullTierName/fullTierName_1998/g' "${base}_1998$ext"
sed -i '.bak' 's/str(year)/\"1998\"/g' "${base}_1998$ext"
sed -i '.bak' 's/audio_feature.csv/audio_feature_1998.csv/g' "${base}_1998$ext"

rm "${base}_1998$ext.bak"
