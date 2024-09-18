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
    using System.Linq;

    /// <summary>
    /// Response when adding a batch of labeled example utterances.
    /// </summary>
    public partial class BatchLabelExample
    {
        /// <summary>
        /// Initializes a new instance of the BatchLabelExample class.
        /// </summary>
        public BatchLabelExample()
        {
            CustomInit();
        }

        /// <summary>
        /// Initializes a new instance of the BatchLabelExample class.
        /// </summary>
        public BatchLabelExample(LabelExampleResponse value = default(LabelExampleResponse), bool? hasError = default(bool?), OperationStatus error = default(OperationStatus))
        {
            Value = value;
            HasError = hasError;
            Error = error;
            CustomInit();
        }

        /// <summary>
        /// An initialization method that performs custom operations like setting defaults
        /// </summary>
        partial void CustomInit();

        /// <summary>
        /// </summary>
        [JsonProperty(PropertyName = "value")]
        public LabelExampleResponse Value { get; set; }

        /// <summary>
        /// </summary>
        [JsonProperty(PropertyName = "hasError")]
        public bool? HasError { get; set; }

        /// <summary>
        /// </summary>
        [JsonProperty(PropertyName = "error")]
        public OperationStatus Error { get; set; }

    }
}
