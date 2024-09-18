// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// <auto-generated/>

#nullable disable

namespace Azure.ResourceManager.CosmosDB.Models
{
    /// <summary> The MongoDBDatabasePropertiesOptions. </summary>
    public partial class MongoDBDatabasePropertiesOptions : OptionsResource
    {
        /// <summary> Initializes a new instance of MongoDBDatabasePropertiesOptions. </summary>
        public MongoDBDatabasePropertiesOptions()
        {
        }

        /// <summary> Initializes a new instance of MongoDBDatabasePropertiesOptions. </summary>
        /// <param name="throughput"> Value of the Cosmos DB resource throughput or autoscaleSettings. Use the ThroughputSetting resource when retrieving offer details. </param>
        /// <param name="autoscaleSettings"> Specifies the Autoscale settings. </param>
        internal MongoDBDatabasePropertiesOptions(int? throughput, AutoscaleSettings autoscaleSettings) : base(throughput, autoscaleSettings)
        {
        }
    }
}
