// Copyright (c) 2012, the Dart project authors.  Please see the AUTHORS file
// for details. All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

// @dart = 2.9

import "package:expect/expect.dart";

main() {
  try {
    Uri base = Uri.base;
    Expect.isTrue(Uri.base.isScheme("file") || Uri.base.isScheme("http"));
  } on UnsupportedError catch (e) {
    Expect.isTrue(e.toString().contains("'Uri.base' is not supported"));
  }
}
