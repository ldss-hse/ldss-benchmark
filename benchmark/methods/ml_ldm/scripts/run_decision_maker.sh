#set -ex
echo $(java --version)

while (( "$#" )); do
  case $1 in
    -JAR_PATH)
      JAR_PATH="$2"
      shift 2
      ;;
    -INPUT_JSON)
      INPUT_JSON="$2"
      shift 2
      ;;
    -OUTPUT_DIR)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    *)
      echo Unsupport argument $1
      exit 1
      ;;
  esac
done

java -jar ${JAR_PATH} -i ${INPUT_JSON} -o ${OUTPUT_DIR}
