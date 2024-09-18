// Copyright (c) 2020, the Dart project authors.  Please see the AUTHORS file
// for details. All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

// @dart = 2.9

// Generated by prepare_flutter_bundle.dart and not ignoring vmspecific.
// Used in fuchsia_test_component.
//
// Note that we are not using any VM flags here, so we are not exercising GC
// corner cases etc. However, we are mainly interested in exercising the
// calling convention on Fuchsia.

import "dart:async";

import "aliasing_test.dart" as main0;
import "data_not_asan_test.dart" as main1;
import "data_test.dart" as main2;
import "dylib_isolates_test.dart" as main3;
import "extension_methods_test.dart" as main4;
import "external_typed_data_test.dart" as main5;
import "function_callbacks_many_test.dart" as main6;
import "function_callbacks_test.dart" as main7;
import "function_callbacks_very_many_test.dart" as main8;
import "function_structs_test.dart" as main9;
import "function_test.dart" as main10;
import "function_very_many_test.dart" as main11;
import "hardfp_test.dart" as main12;
import "negative_function_test.dart" as main13;
import "null_regress_39068_test.dart" as main14;
import "null_test.dart" as main15;
import "regress_37254_test.dart" as main16;
import "regress_39044_test.dart" as main17;
import "regress_39063_test.dart" as main18;
import "regress_39885_test.dart" as main19;
import "regress_40537_test.dart" as main20;
import "regress_43016_test.dart" as main21;
import "regress_43693_test.dart" as main22;
import "regress_jump_to_frame_test.dart" as main23;
import "sizeof_test.dart" as main24;
import "snapshot_test.dart" as main25;
import "stacktrace_regress_37910_test.dart" as main26;
import "structs_test.dart" as main27;
import "variance_function_test.dart" as main28;
import "vmspecific_function_callbacks_exit_test.dart" as main29;
import "vmspecific_function_test.dart" as main30;
import "vmspecific_handle_dynamically_linked_test.dart" as main31;
import "vmspecific_handle_test.dart" as main32;
import "vmspecific_null_test.dart" as main33;
import "vmspecific_object_gc_test.dart" as main34;
import "vmspecific_regress_37100_test.dart" as main35;
import "vmspecific_regress_37511_callbacks_test.dart" as main36;
import "vmspecific_regress_37511_test.dart" as main37;
import "vmspecific_regress_37780_test.dart" as main38;

Future invoke(dynamic fun) async {
  if (fun is void Function() || fun is Future Function()) {
    return await fun();
  } else {
    return await fun(<String>[]);
  }
}

dynamic main() async {
  await invoke(main0.main);
  await invoke(main1.main);
  await invoke(main2.main);
  await invoke(main3.main);
  await invoke(main4.main);
  await invoke(main5.main);
  await invoke(main6.main);
  await invoke(main7.main);
  await invoke(main8.main);
  await invoke(main9.main);
  await invoke(main10.main);
  await invoke(main11.main);
  await invoke(main12.main);
  await invoke(main13.main);
  await invoke(main14.main);
  await invoke(main15.main);
  await invoke(main16.main);
  await invoke(main17.main);
  await invoke(main18.main);
  await invoke(main19.main);
  await invoke(main20.main);
  await invoke(main21.main);
  await invoke(main22.main);
  await invoke(main23.main);
  await invoke(main24.main);
  await invoke(main25.main);
  await invoke(main26.main);
  await invoke(main27.main);
  await invoke(main28.main);
  await invoke(main29.main);
  await invoke(main30.main);
  await invoke(main31.main);
  await invoke(main32.main);
  await invoke(main33.main);
  await invoke(main34.main);
  await invoke(main35.main);
  await invoke(main36.main);
  await invoke(main37.main);
  await invoke(main38.main);
}
