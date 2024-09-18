// <auto-generated>
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for
// license information.
//
// Code generated by Microsoft (R) AutoRest Code Generator.
// Changes may cause incorrect behavior and will be lost if the code is
// regenerated.
// </auto-generated>

namespace Microsoft.Azure.Management.Network.Models
{
    using Newtonsoft.Json;
    using System.Collections;
    using System.Collections.Generic;
    using System.Linq;

    /// <summary>
    /// DNS Proxy Settings in Firewall Policy.
    /// </summary>
    public partial class DnsSettings
    {
        /// <summary>
        /// Initializes a new instance of the DnsSettings class.
        /// </summary>
        public DnsSettings()
        {
            CustomInit();
        }

        /// <summary>
        /// Initializes a new instance of the DnsSettings class.
        /// </summary>
        /// <param name="servers">List of Custom DNS Servers.</param>
        /// <param name="enableProxy">Enable DNS Proxy on Firewalls attached to
        /// the Firewall Policy.</param>
        /// <param name="requireProxyForNetworkRules">FQDNs in Network Rules
        /// are supported when set to true.</param>
        public DnsSettings(IList<string> servers = default(IList<string>), bool? enableProxy = default(bool?), bool? requireProxyForNetworkRules = default(bool?))
        {
            Servers = servers;
            EnableProxy = enableProxy;
            RequireProxyForNetworkRules = requireProxyForNetworkRules;
            CustomInit();
        }

        /// <summary>
        /// An initialization method that performs custom operations like setting defaults
        /// </summary>
        partial void CustomInit();

        /// <summary>
        /// Gets or sets list of Custom DNS Servers.
        /// </summary>
        [JsonProperty(PropertyName = "servers")]
        public IList<string> Servers { get; set; }

        /// <summary>
        /// Gets or sets enable DNS Proxy on Firewalls attached to the Firewall
        /// Policy.
        /// </summary>
        [JsonProperty(PropertyName = "enableProxy")]
        public bool? EnableProxy { get; set; }

        /// <summary>
        /// Gets or sets fQDNs in Network Rules are supported when set to true.
        /// </summary>
        [JsonProperty(PropertyName = "requireProxyForNetworkRules")]
        public bool? RequireProxyForNetworkRules { get; set; }

    }
}
