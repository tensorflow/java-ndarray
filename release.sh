#!/usr/bin/env bash
# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Script to upload release artifacts for the TensorFlow Java library to
# Maven Central. See RELEASE.md for explanation.

cd $(dirname "$0")
shift
shift
CMD="$*"

# If release fails, debug with
#   ./release.sh bash
# To get a shell to poke around the maven artifacts with.
if [[ -z "${CMD}" ]]
then
  CMD="mvn clean deploy -B -e --settings ./settings.xml -Preleasing"
fi

export GPG_TTY=$(tty)
set -ex

if [[ ! -f settings.xml ]]
then
  cp -f ~/.m2/settings.xml .
fi

docker run \
  -e GPG_TTY="${GPG_TTY}" \
  -v ${PWD}:/tensorflow-java-ndarray \
  -v ${HOME}/.gnupg:/root/.gnupg \
  -w /tensorflow-java-ndarray \
  -it \
  --platform linux/amd64 \
  maven:3.8.6-jdk-11  \
  ${CMD}

echo
echo "Release completed"
