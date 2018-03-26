sh "generate_all_codefiles.sh"

find . -type f -regex '.*/audio_feature_[0-9].*' > codefiles

while IFS= read -r line;do
    echo -n "python " >> final_run.sh
    echo -n "$line" >> final_run.sh
    echo " &" >> final_run.sh
done < "codefiles"

echo "wait" >> final_run.sh

echo "completed"

rm codefiles
