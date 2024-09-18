// <auto-generated>
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for
// license information.
//
// Code generated by Microsoft (R) AutoRest Code Generator.
// Changes may cause incorrect behavior and will be lost if the code is
// regenerated.
// </auto-generated>

namespace Microsoft.Azure.Management.ContainerRegistry.Models
{
    using Newtonsoft.Json;
    using System.Linq;

    /// <summary>
    /// Describes the properties of a secret object value.
    /// </summary>
    public partial class SecretObject
    {
        /// <summary>
        /// Initializes a new instance of the SecretObject class.
        /// </summary>
        public SecretObject()
        {
            CustomInit();
        }

        /// <summary>
        /// Initializes a new instance of the SecretObject class.
        /// </summary>
        /// <param name="value">The value of the secret. The format of this
        /// value will be determined
        /// based on the type of the secret object. If the type is Opaque, the
        /// value will be
        /// used as is without any modification.</param>
        /// <param name="type">The type of the secret object which determines
        /// how the value of the secret object has to be
        /// interpreted. Possible values include: 'Opaque',
        /// 'Vaultsecret'</param>
        public SecretObject(string value = default(string), string type = default(string))
        {
            Value = value;
            Type = type;
            CustomInit();
        }

        /// <summary>
        /// An initialization method that performs custom operations like setting defaults
        /// </summary>
        partial void CustomInit();

        /// <summary>
        /// Gets or sets the value of the secret. The format of this value will
        /// be determined
        /// based on the type of the secret object. If the type is Opaque, the
        /// value will be
        /// used as is without any modification.
        /// </summary>
        [JsonProperty(PropertyName = "value")]
        public string Value { get; set; }

        /// <summary>
        /// Gets or sets the type of the secret object which determines how the
        /// value of the secret object has to be
        /// interpreted. Possible values include: 'Opaque', 'Vaultsecret'
        /// </summary>
        [JsonProperty(PropertyName = "type")]
        public string Type { get; set; }

    }
}
