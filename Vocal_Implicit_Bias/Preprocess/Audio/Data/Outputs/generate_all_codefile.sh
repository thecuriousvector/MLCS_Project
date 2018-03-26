base="audio_feature"
ext=".py"

sh "generate_code.sh"

for yr in 1999 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012
do
  cp "generate_code.sh" "generate_code_tmp${yr}.sh"
  sed -i '.bak' "s/1998/$yr/g" "generate_code_tmp${yr}.sh"
  chmod 777 "generate_code_tmp${yr}.sh"
  sh "generate_code_tmp${yr}.sh"
  rm "generate_code_tmp${yr}.sh"
  rm "generate_code_tmp${yr}.sh.bak"
done
