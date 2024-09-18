/*
 * Copyright 2010-2021 JetBrains s.r.o. and Kotlin Programming Language contributors.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the license/LICENSE.txt file.
 */

package org.jetbrains.kotlin.analysis.api.fe10.components.expressionInfoProvider;

import com.intellij.testFramework.TestDataPath;
import org.jetbrains.kotlin.test.util.KtTestUtil;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.kotlin.analysis.api.impl.barebone.test.FrontendApiTestConfiguratorService;
import org.jetbrains.kotlin.analysis.api.descriptors.test.KtFe10FrontendApiTestConfiguratorService;
import org.jetbrains.kotlin.analysis.api.impl.base.test.components.expressionInfoProvider.AbstractWhenMissingCasesTest;
import org.jetbrains.kotlin.test.TestMetadata;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.util.regex.Pattern;

/** This class is generated by {@link GenerateNewCompilerTests.kt}. DO NOT MODIFY MANUALLY */
@SuppressWarnings("all")
@TestMetadata("analysis/analysis-api/testData/components/expressionInfoProvider/whenMissingCases")
@TestDataPath("$PROJECT_ROOT")
public class Fe10WhenMissingCasesTestGenerated extends AbstractWhenMissingCasesTest {
    @NotNull
    @Override
    public FrontendApiTestConfiguratorService getConfigurator() {
        return KtFe10FrontendApiTestConfiguratorService.INSTANCE;
    }

    @Test
    public void testAllFilesPresentInWhenMissingCases() throws Exception {
        KtTestUtil.assertAllTestsPresentByMetadataWithExcluded(this.getClass(), new File("analysis/analysis-api/testData/components/expressionInfoProvider/whenMissingCases"), Pattern.compile("^(.+)\\.kt$"), null, true);
    }

    @Test
    @TestMetadata("boolean_else.kt")
    public void testBoolean_else() throws Exception {
        runTest("analysis/analysis-api/testData/components/expressionInfoProvider/whenMissingCases/boolean_else.kt");
    }

    @Test
    @TestMetadata("boolean_empty.kt")
    public void testBoolean_empty() throws Exception {
        runTest("analysis/analysis-api/testData/components/expressionInfoProvider/whenMissingCases/boolean_empty.kt");
    }

    @Test
    @TestMetadata("boolean_partial.kt")
    public void testBoolean_partial() throws Exception {
        runTest("analysis/analysis-api/testData/components/expressionInfoProvider/whenMissingCases/boolean_partial.kt");
    }

    @Test
    @TestMetadata("enum_else.kt")
    public void testEnum_else() throws Exception {
        runTest("analysis/analysis-api/testData/components/expressionInfoProvider/whenMissingCases/enum_else.kt");
    }

    @Test
    @TestMetadata("enum_empty.kt")
    public void testEnum_empty() throws Exception {
        runTest("analysis/analysis-api/testData/components/expressionInfoProvider/whenMissingCases/enum_empty.kt");
    }

    @Test
    @TestMetadata("enum_partial.kt")
    public void testEnum_partial() throws Exception {
        runTest("analysis/analysis-api/testData/components/expressionInfoProvider/whenMissingCases/enum_partial.kt");
    }

    @Test
    @TestMetadata("nothing.kt")
    public void testNothing() throws Exception {
        runTest("analysis/analysis-api/testData/components/expressionInfoProvider/whenMissingCases/nothing.kt");
    }

    @Test
    @TestMetadata("nullableBoolean.kt")
    public void testNullableBoolean() throws Exception {
        runTest("analysis/analysis-api/testData/components/expressionInfoProvider/whenMissingCases/nullableBoolean.kt");
    }

    @Test
    @TestMetadata("nullableEnum.kt")
    public void testNullableEnum() throws Exception {
        runTest("analysis/analysis-api/testData/components/expressionInfoProvider/whenMissingCases/nullableEnum.kt");
    }

    @Test
    @TestMetadata("nullableNothing.kt")
    public void testNullableNothing() throws Exception {
        runTest("analysis/analysis-api/testData/components/expressionInfoProvider/whenMissingCases/nullableNothing.kt");
    }

    @Test
    @TestMetadata("nullableSealedClass_empty.kt")
    public void testNullableSealedClass_empty() throws Exception {
        runTest("analysis/analysis-api/testData/components/expressionInfoProvider/whenMissingCases/nullableSealedClass_empty.kt");
    }

    @Test
    @TestMetadata("sealedClass_else.kt")
    public void testSealedClass_else() throws Exception {
        runTest("analysis/analysis-api/testData/components/expressionInfoProvider/whenMissingCases/sealedClass_else.kt");
    }

    @Test
    @TestMetadata("sealedClass_empty.kt")
    public void testSealedClass_empty() throws Exception {
        runTest("analysis/analysis-api/testData/components/expressionInfoProvider/whenMissingCases/sealedClass_empty.kt");
    }

    @Test
    @TestMetadata("sealedClass_partial.kt")
    public void testSealedClass_partial() throws Exception {
        runTest("analysis/analysis-api/testData/components/expressionInfoProvider/whenMissingCases/sealedClass_partial.kt");
    }
}
