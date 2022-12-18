#!/bin/bash

function display_help {
    echo 'USAGE:'
    echo 'docker run -ti [--rm] --mount source=$(pwd)/src,destination=/tmp/echr_experiments,type=bind,readonly <image_id|image_name[:image_tag]> [help | bash | /bin/bash | deploy | build | pytest [pytest options] [file_or_dir] [...] | lint | all]'
    echo
    echo 'OPTIONS:'
    echo '  help              - Prints this help and exits.'
    echo '  bash | /bin/bash  - Allows to access bash console of the container.'
    echo '  test              - Runs pytest'
    echo '  run               - Run the experiments'
    echo '  analyze           - Analyze the results of the experiments'
    echo '  lint              - Runs pylint.'
}

function check {
    if [[ "$2" = 'binary' ]] ; then
        python3 ./binary_check.py ${@:3}
    elif [[ "$2" = "multiclass" ]] ; then
        python3 ./multiclass_check.py ${@:3}
    elif [[ "$2" = "multilabel" ]] ; then
        python3 ./multilabel_check.py ${@:3}
    fi
}

function run {
    if [[ "$2" = 'binary' ]] ; then
        python3 ./binary_experiments.py ${@:3}
    elif [[ "$2" = "multiclass" ]] ; then
        python3 ./multiclass_experiments.py ${@:3}
    elif [[ "$2" = "multilabel" ]] ; then
        python3 ./multilabel_experiments.py ${@:3}
    fi
}

function analyze {
    if [[ "$2" = 'binary' ]] ; then
        echo "Confusion Matrices"
        python3 ./binary_confusion_matrices.py
        echo "Latex Tables"
        python3 ./binary_generate_latex.py
    elif [[ "$2" = "multiclass" ]] ; then
        echo "Confusion Matrices"
        python3 ./multiclass_confusion_matrices.py
        echo "Latex Tables"
        python3 ./multiclass_generate_latex.py
    elif [[ "$2" = "multilabel" ]] ; then
        echo "Latex Tables"
        python3 ./multilabel_generate_latex.py
    fi
}

function reports {
    python3 ./generate_reports.py
    pdflatex ./data/analysis/report.tex  
    pdflatex ./data/analysis/report.tex  
    mv ./report.pdf ./data/analysis/report.pdf
}

function lint_source_code {
    python -m pylint --rcfile=.pylintrc *.py
}

function handle_input {
    if [[ "$#" -eq 0 ]] ; then
        display_help
    else
        if [[ "$1" = 'bash' || "$1" = '/bin/bash' ]] ; then
            /bin/bash
        elif [[ "$1" = "run" ]] ; then
          run $@
        elif [[ "$1" = "check" ]] ; then
          check $@
        elif [[ "$1" = "analyze" ]] ; then
          analyze $@
        elif [[ "$1" = "reports" ]] ; then
          reports $@
        elif [[ "$1" = "test" ]] ; then
            python -m pytest -v -c ./.pytest.ini --disable-warnings &&\
            lint_source_code
        elif [[ "$1" = 'lint' ]] ; then
            lint_source_code
        else
            display_help
        fi
    fi
}

function main() {
    handle_input $@
    status_code=$?
    exit ${status_code}
}

main $@