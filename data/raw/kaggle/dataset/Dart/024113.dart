// Copyright (c) 2022, the Dart project authors.  Please see the AUTHORS file
// for details. All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

class C {
  String log = "";

  C(int x, int y, {int z = 42}) {
    log = "x=$x, y=$y, z=$z";
  }

  C.named1(int x, int y, int z) : this(x, y, z: z);
  C.named2(int x, int y, int z) : this(x, z: z, y);
  C.named3(int x, int y, int z) : this(z: z, x, y);
}

main() {
  expect("x=1, y=2, z=3", C(1, 2, z: 3).log);
  expect("x=1, y=2, z=3", C(z: 3, 1, 2).log);
  expect("x=1, y=2, z=3", C(1, z: 3, 2).log);
  expect("x=1, y=2, z=3", C.named1(1, 2, 3).log);
  expect("x=1, y=2, z=3", C.named2(1, 2, 3).log);
  expect("x=1, y=2, z=3", C.named3(1, 2, 3).log);
}

expect(expected, actual) {
  if (expected != actual) throw 'Expected $expected, actual $actual';
}