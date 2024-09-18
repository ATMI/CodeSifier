// Copyright (c) 2021, the Dart project authors. Please see the AUTHORS file
// for details. All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

import 'package:analyzer/src/error/codes.dart';
import 'package:analyzer/src/test_utilities/find_element.dart';
import 'package:test_reflective_loader/test_reflective_loader.dart';

import 'context_collection_resolution.dart';

main() {
  defineReflectiveSuite(() {
    defineReflectiveTests(SuperFormalParameterTest);
  });
}

@reflectiveTest
class SuperFormalParameterTest extends PubPackageResolutionTest {
  test_functionTyped() async {
    await assertNoErrorsInCode(r'''
class A {
  A(Object a);
}

class B extends A {
  B(super.a<T>(int b));
}
''');

    var B = findElement.unnamedConstructor('B');
    var element = B.superFormalParameter('a');

    assertElement(
      findNode.superFormalParameter('super.a'),
      element,
    );

    assertElement(
      findNode.typeParameter('T>'),
      element.typeParameters[0],
    );

    assertElement(
      findNode.simpleFormalParameter('b));'),
      element.parameters[0],
    );
  }

  test_invalid_notConstructor() async {
    await assertErrorsInCode(r'''
void f(super.a) {}
''', [
      error(CompileTimeErrorCode.INVALID_SUPER_FORMAL_PARAMETER_LOCATION, 7, 5),
    ]);

    var f = findElement.topFunction('f');
    var element = f.superFormalParameter('a');
    assertTypeDynamic(element.type);

    assertElement(
      findNode.superFormalParameter('super.a'),
      element,
    );
  }

  test_optionalNamed() async {
    await assertNoErrorsInCode(r'''
class A {
  A({int? a});
}

class B extends A {
  B({super.a});
}
''');

    assertElement(
      findNode.superFormalParameter('super.a'),
      findElement.unnamedConstructor('B').superFormalParameter('a'),
    );
  }

  test_optionalPositional() async {
    await assertNoErrorsInCode(r'''
class A {
  A([int? a]);
}

class B extends A {
  B([super.a]);
}
''');

    assertElement(
      findNode.superFormalParameter('super.a'),
      findElement.unnamedConstructor('B').superFormalParameter('a'),
    );
  }

  test_requiredNamed() async {
    await assertNoErrorsInCode(r'''
class A {
  A({required int a});
}

class B extends A {
  B({required super.a});
}
''');

    assertElement(
      findNode.superFormalParameter('super.a'),
      findElement.unnamedConstructor('B').superFormalParameter('a'),
    );
  }

  test_requiredPositional() async {
    await assertNoErrorsInCode(r'''
class A {
  A(int a);
}

class B extends A {
  B(super.a);
}
''');

    assertElement(
      findNode.superFormalParameter('super.a'),
      findElement.unnamedConstructor('B').superFormalParameter('a'),
    );
  }

  test_scoping_inBody() async {
    await assertNoErrorsInCode(r'''
class A {
  final int a;
  A(this.a);
}

class B extends A {
  B(super.a) {
    a; // ref
  }
}
''');

    assertElement(
      findNode.simple('a; // ref'),
      findElement.getter('a', of: 'A'),
    );
  }

  test_scoping_inInitializer() async {
    await assertNoErrorsInCode(r'''
class A {
  A(int a);
}

class B extends A {
  var f;
  B(super.a) : f = ((){ a; });
}
''');

    assertElement(
      findNode.simple('a; }'),
      findElement.unnamedConstructor('B').superFormalParameter('a'),
    );
  }
}
