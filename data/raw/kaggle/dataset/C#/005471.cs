// <auto-generated>
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for
// license information.
//
// Code generated by Microsoft (R) AutoRest Code Generator.
// Changes may cause incorrect behavior and will be lost if the code is
// regenerated.
// </auto-generated>

namespace Microsoft.Azure.Management.DataFactory.Models
{
    using Microsoft.Rest;
    using Microsoft.Rest.Serialization;
    using Newtonsoft.Json;
    using System.Collections;
    using System.Collections.Generic;
    using System.Linq;

    /// <summary>
    /// The Azure Data Lake Storage Gen2 storage.
    /// </summary>
    [Newtonsoft.Json.JsonObject("AzureBlobFSFile")]
    [Rest.Serialization.JsonTransformation]
    public partial class AzureBlobFSDataset : Dataset
    {
        /// <summary>
        /// Initializes a new instance of the AzureBlobFSDataset class.
        /// </summary>
        public AzureBlobFSDataset()
        {
            LinkedServiceName = new LinkedServiceReference();
            CustomInit();
        }

        /// <summary>
        /// Initializes a new instance of the AzureBlobFSDataset class.
        /// </summary>
        /// <param name="linkedServiceName">Linked service reference.</param>
        /// <param name="additionalProperties">Unmatched properties from the
        /// message are deserialized this collection</param>
        /// <param name="description">Dataset description.</param>
        /// <param name="structure">Columns that define the structure of the
        /// dataset. Type: array (or Expression with resultType array),
        /// itemType: DatasetDataElement.</param>
        /// <param name="schema">Columns that define the physical type schema
        /// of the dataset. Type: array (or Expression with resultType array),
        /// itemType: DatasetSchemaDataElement.</param>
        /// <param name="parameters">Parameters for dataset.</param>
        /// <param name="annotations">List of tags that can be used for
        /// describing the Dataset.</param>
        /// <param name="folder">The folder that this Dataset is in. If not
        /// specified, Dataset will appear at the root level.</param>
        /// <param name="folderPath">The path of the Azure Data Lake Storage
        /// Gen2 storage. Type: string (or Expression with resultType
        /// string).</param>
        /// <param name="fileName">The name of the Azure Data Lake Storage
        /// Gen2. Type: string (or Expression with resultType string).</param>
        /// <param name="format">The format of the Azure Data Lake Storage Gen2
        /// storage.</param>
        /// <param name="compression">The data compression method used for the
        /// blob storage.</param>
        public AzureBlobFSDataset(LinkedServiceReference linkedServiceName, IDictionary<string, object> additionalProperties = default(IDictionary<string, object>), string description = default(string), object structure = default(object), object schema = default(object), IDictionary<string, ParameterSpecification> parameters = default(IDictionary<string, ParameterSpecification>), IList<object> annotations = default(IList<object>), DatasetFolder folder = default(DatasetFolder), object folderPath = default(object), object fileName = default(object), DatasetStorageFormat format = default(DatasetStorageFormat), DatasetCompression compression = default(DatasetCompression))
            : base(linkedServiceName, additionalProperties, description, structure, schema, parameters, annotations, folder)
        {
            FolderPath = folderPath;
            FileName = fileName;
            Format = format;
            Compression = compression;
            CustomInit();
        }

        /// <summary>
        /// An initialization method that performs custom operations like setting defaults
        /// </summary>
        partial void CustomInit();

        /// <summary>
        /// Gets or sets the path of the Azure Data Lake Storage Gen2 storage.
        /// Type: string (or Expression with resultType string).
        /// </summary>
        [JsonProperty(PropertyName = "typeProperties.folderPath")]
        public object FolderPath { get; set; }

        /// <summary>
        /// Gets or sets the name of the Azure Data Lake Storage Gen2. Type:
        /// string (or Expression with resultType string).
        /// </summary>
        [JsonProperty(PropertyName = "typeProperties.fileName")]
        public object FileName { get; set; }

        /// <summary>
        /// Gets or sets the format of the Azure Data Lake Storage Gen2
        /// storage.
        /// </summary>
        [JsonProperty(PropertyName = "typeProperties.format")]
        public DatasetStorageFormat Format { get; set; }

        /// <summary>
        /// Gets or sets the data compression method used for the blob storage.
        /// </summary>
        [JsonProperty(PropertyName = "typeProperties.compression")]
        public DatasetCompression Compression { get; set; }

        /// <summary>
        /// Validate the object.
        /// </summary>
        /// <exception cref="ValidationException">
        /// Thrown if validation fails
        /// </exception>
        public override void Validate()
        {
            base.Validate();
            if (Compression != null)
            {
                Compression.Validate();
            }
        }
    }
}
