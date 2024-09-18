// <auto-generated>
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for
// license information.
//
// Code generated by Microsoft (R) AutoRest Code Generator.
// Changes may cause incorrect behavior and will be lost if the code is
// regenerated.
// </auto-generated>

namespace Microsoft.Azure.CognitiveServices.Language.LUIS.Authoring.Models
{
    using Newtonsoft.Json;
    using System.Collections;
    using System.Collections.Generic;
    using System.Linq;

    /// <summary>
    /// Model Features, including Patterns and Phraselists.
    /// </summary>
    public partial class FeaturesResponseObject
    {
        /// <summary>
        /// Initializes a new instance of the FeaturesResponseObject class.
        /// </summary>
        public FeaturesResponseObject()
        {
            CustomInit();
        }

        /// <summary>
        /// Initializes a new instance of the FeaturesResponseObject class.
        /// </summary>
        public FeaturesResponseObject(IList<PhraseListFeatureInfo> phraselistFeatures = default(IList<PhraseListFeatureInfo>), IList<PatternFeatureInfo> patternFeatures = default(IList<PatternFeatureInfo>))
        {
            PhraselistFeatures = phraselistFeatures;
            PatternFeatures = patternFeatures;
            CustomInit();
        }

        /// <summary>
        /// An initialization method that performs custom operations like setting defaults
        /// </summary>
        partial void CustomInit();

        /// <summary>
        /// </summary>
        [JsonProperty(PropertyName = "phraselistFeatures")]
        public IList<PhraseListFeatureInfo> PhraselistFeatures { get; set; }

        /// <summary>
        /// </summary>
        [JsonProperty(PropertyName = "patternFeatures")]
        public IList<PatternFeatureInfo> PatternFeatures { get; set; }

    }
}
