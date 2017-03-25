FOLDER=code
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
NAME=${DIR##*/}
rsync -rP \
--exclude "/$NAME/data" \
--exclude "/$NAME/results" \
--exclude "/$NAME/logs" \
--exclude ".git*" \
--exclude "*.pyc" \
../$NAME $USER@cs.toronto.edu:/u/$USER/$FOLDER 