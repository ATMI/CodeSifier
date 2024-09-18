// Copyright (c) 2021, the Dart project authors.  Please see the AUTHORS file
// for details. All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

// @dart = 2.10

/*library: 
 a_pre_fragments=[
  p1: {units: [6{b4}], usedBy: [], needs: []},
  p2: {units: [4{b2}], usedBy: [], needs: []},
  p3: {units: [1{b1}], usedBy: [], needs: []},
  p4: {units: [5{b2, b4}], usedBy: [], needs: []},
  p5: {units: [3{b1, b4}], usedBy: [], needs: []},
  p6: {units: [2{b1, b2, b3, b4}], usedBy: [], needs: []}],
 b_finalized_fragments=[
  f1: [6{b4}],
  f2: [4{b2}],
  f3: [1{b1}],
  f4: [5{b2, b4}],
  f5: [3{b1, b4}],
  f6: [2{b1, b2, b3, b4}]],
 c_steps=[
  b1=(f6, f5, f3),
  b2=(f6, f4, f2),
  b3=(f6),
  b4=(f6, f5, f4, f1)]
*/

// This file was autogenerated by the pkg/compiler/tool/graph_isomorphizer.dart.
import 'lib1.dart';
import 'lib2.dart';
import 'lib3.dart';
import 'lib4.dart';

/*member: main:member_unit=main{}*/
main() {
  entryLib1();
  entryLib2();
  entryLib3();
  entryLib4();
}
