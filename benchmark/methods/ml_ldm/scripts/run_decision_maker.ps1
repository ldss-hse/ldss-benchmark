param ($JAR_PATH, $INPUT_JSON, $OUTPUT_DIR)

$currentDirectory = Get-Location;
Write-Output "Current directory: $currentDirectory";

Write-Output "Script root: $PSScriptRoot";

Write-Output "JAR_PATH: $JAR_PATH";
Write-Output "INPUT_JSON: $INPUT_JSON";
Write-Output "OUTPUT_TXT: $OUTPUT_DIR";

if ($JAR_PATH -eq $null) {
    "Environment variables do not exist, filling them with defaults";
    $JAR_PATH = $PSScriptRoot + "/bin/lingvo-dss-all.jar";
    $INPUT_JSON = $PSScriptRoot + "/bin/description_multilevel.json";
    $OUTPUT_DIR = $PSScriptRoot + "/../artifacts/";
}

& java -jar ${JAR_PATH} -i $INPUT_JSON -o $OUTPUT_DIR
