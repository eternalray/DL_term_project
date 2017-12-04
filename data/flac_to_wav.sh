find . -type f -name '*.flac' -exec sh -c '
orgfile="$0"
newfile="$(echo "$0" | rev | cut -d"/" -f1 | rev | sed 's/.flac/.wav/')"
echo $newfile
ffmpeg -i $orgfile ../../save/$newfile
' {} \;