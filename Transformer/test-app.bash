#!/bin/bash

docker run --rm --link sonarqube -e SONAR_HOST_URL="http://sonarqube:9000" -e SONAR_LOGIN="sqp_869434287bdeff4ded3da89be003d83b80fc53a9" -v ${PWD}:/usr/src sonarsource/sonar-scanner-cli