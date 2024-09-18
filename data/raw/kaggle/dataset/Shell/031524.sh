#!/usr/bin/env bash
# Uploads a new version of the co19_2 CIPD package.
# This script requires access to the dart-build-access group, which EngProd has.

set -e
set -x

if [ ! -e tests/co19_2 ]; then
  echo "$0: error: Run this script at the root of the Dart SDK" >&2
  exit 1
fi

# Find the latest co19 commit.
rm -rf tests/co19_2/src.git
git clone https://dart.googlesource.com/co19 tests/co19_2/src.git
CO19=tests/co19_2/src.git
OLD=$(gclient getdep --var=co19_2_rev)
NEW=$(cd $CO19 && git fetch origin && git rev-parse origin/pre-nnbd)

git fetch origin
git branch cl-co19-roll-co19-to-$NEW origin/main
git checkout cl-co19-roll-co19-to-$NEW

# Update DEPS:
gclient setdep --var=co19_2_rev=$NEW

BUILDERS=$(jq -r '.builder_configurations
  | map(select(.steps
    | any(.arguments
      | select(.!=null)
        | any(test("co19_2($|(/.*))")))))
  | map(.builders)
  | flatten
  | sort
  | .[] += "-try"
  | join(",")' \
  tools/bots/test_matrix.json)

# Make a nice commit. Don't include the '#' character to avoid referencing Dart
# SDK issues.
git commit DEPS -m \
  "$(printf "[co19] Roll co19_2 to $NEW\n\n" \
  && cd $CO19 \
  && git log --date='format:%Y-%m-%d' --pretty='format:%ad %ae %s' $OLD..$NEW \
    | sed 's/\#/dart-lang\/co19\#/g' \
  && printf "\nCq-Include-Trybots: dart/try:$BUILDERS\n")"

rm -rf tests/co19_2/src.git

GIT_EDITOR=true git cl upload
ISSUE=$(git config --get branch.cl-co19-roll-co19-to-$NEW.gerritissue)

git cl web

set +x
cat << EOF

Wait for the builders to finish. If any failed, pre-approve them.
EOF
