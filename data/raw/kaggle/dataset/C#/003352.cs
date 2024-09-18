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
    using System.Collections;
    using System.Collections.Generic;
    using System.Linq;

    /// <summary>
    /// The response from the GenerateCredentials operation.
    /// </summary>
    public partial class GenerateCredentialsResult
    {
        /// <summary>
        /// Initializes a new instance of the GenerateCredentialsResult class.
        /// </summary>
        public GenerateCredentialsResult()
        {
            CustomInit();
        }

        /// <summary>
        /// Initializes a new instance of the GenerateCredentialsResult class.
        /// </summary>
        /// <param name="username">The username for a container
        /// registry.</param>
        /// <param name="passwords">The list of passwords for a container
        /// registry.</param>
        public GenerateCredentialsResult(string username = default(string), IList<TokenPassword> passwords = default(IList<TokenPassword>))
        {
            Username = username;
            Passwords = passwords;
            CustomInit();
        }

        /// <summary>
        /// An initialization method that performs custom operations like setting defaults
        /// </summary>
        partial void CustomInit();

        /// <summary>
        /// Gets or sets the username for a container registry.
        /// </summary>
        [JsonProperty(PropertyName = "username")]
        public string Username { get; set; }

        /// <summary>
        /// Gets or sets the list of passwords for a container registry.
        /// </summary>
        [JsonProperty(PropertyName = "passwords")]
        public IList<TokenPassword> Passwords { get; set; }

    }
}
