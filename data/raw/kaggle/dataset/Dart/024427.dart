// Copyright (c) 2020, the Dart project authors.  Please see the AUTHORS file
// for details. All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

// @dart=2.9

typedef E1<T> = void Function();
typedef E2<T extends num> = void Function();
typedef E3<T, S> = void Function();
typedef E4<T extends num, S extends num> = void Function();
typedef E5<T extends S, S extends num> = void Function();
typedef E6<T extends num, S extends T> = void Function();

typedef F1 = void Function<T>();
typedef F2 = void Function<T extends num>();
typedef F3 = void Function<T, S>();
typedef F4 = void Function<T extends num, S extends num>();
typedef F5 = void Function<T extends S, S extends num>();
typedef F6 = void Function<T extends num, S extends T>();

typedef G1<X> = void Function<T extends X>();
typedef G2<X extends num> = void Function<T extends X>();
typedef G3<X, Y> = void Function<T extends X, S extends Y>();
typedef G4<X extends num, Y extends num> = void
    Function<T extends X, S extends Y>();
typedef G5<X extends num> = void Function<T extends S, S extends X>();
typedef G6<X extends num> = void Function<T extends X, S extends T>();

typedef H1 = void Function(void Function<T>());
typedef H2 = void Function(void Function<T extends num>());
typedef H3 = void Function(void Function<T, S>());
typedef H4 = void Function(void Function<T extends num, S extends num>());
typedef H5 = void Function(void Function<T extends S, S extends num>());
typedef H6 = void Function(void Function<T extends num, S extends T>());

void Function<T>() f1;
void Function<T extends num>() f2;
void Function<T, S>() f3;
void Function<T extends num, S extends num>() f4;
void Function<T extends S, S extends num>() f5;
void Function<T extends num, S extends T>() f6;

main() {}
