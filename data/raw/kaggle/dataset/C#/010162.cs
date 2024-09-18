// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// <auto-generated/>

#nullable disable

using System;
using System.ComponentModel;

namespace Azure.ResourceManager.Network.Models
{
    /// <summary> Gateway connection type. </summary>
    public readonly partial struct VirtualNetworkGatewayConnectionType : IEquatable<VirtualNetworkGatewayConnectionType>
    {
        private readonly string _value;

        /// <summary> Initializes a new instance of <see cref="VirtualNetworkGatewayConnectionType"/>. </summary>
        /// <exception cref="ArgumentNullException"> <paramref name="value"/> is null. </exception>
        public VirtualNetworkGatewayConnectionType(string value)
        {
            _value = value ?? throw new ArgumentNullException(nameof(value));
        }

        private const string IPsecValue = "IPsec";
        private const string Vnet2VnetValue = "Vnet2Vnet";
        private const string ExpressRouteValue = "ExpressRoute";
        private const string VpnClientValue = "VPNClient";

        /// <summary> IPsec. </summary>
        public static VirtualNetworkGatewayConnectionType IPsec { get; } = new VirtualNetworkGatewayConnectionType(IPsecValue);
        /// <summary> Vnet2Vnet. </summary>
        public static VirtualNetworkGatewayConnectionType Vnet2Vnet { get; } = new VirtualNetworkGatewayConnectionType(Vnet2VnetValue);
        /// <summary> ExpressRoute. </summary>
        public static VirtualNetworkGatewayConnectionType ExpressRoute { get; } = new VirtualNetworkGatewayConnectionType(ExpressRouteValue);
        /// <summary> VPNClient. </summary>
        public static VirtualNetworkGatewayConnectionType VpnClient { get; } = new VirtualNetworkGatewayConnectionType(VpnClientValue);
        /// <summary> Determines if two <see cref="VirtualNetworkGatewayConnectionType"/> values are the same. </summary>
        public static bool operator ==(VirtualNetworkGatewayConnectionType left, VirtualNetworkGatewayConnectionType right) => left.Equals(right);
        /// <summary> Determines if two <see cref="VirtualNetworkGatewayConnectionType"/> values are not the same. </summary>
        public static bool operator !=(VirtualNetworkGatewayConnectionType left, VirtualNetworkGatewayConnectionType right) => !left.Equals(right);
        /// <summary> Converts a string to a <see cref="VirtualNetworkGatewayConnectionType"/>. </summary>
        public static implicit operator VirtualNetworkGatewayConnectionType(string value) => new VirtualNetworkGatewayConnectionType(value);

        /// <inheritdoc />
        [EditorBrowsable(EditorBrowsableState.Never)]
        public override bool Equals(object obj) => obj is VirtualNetworkGatewayConnectionType other && Equals(other);
        /// <inheritdoc />
        public bool Equals(VirtualNetworkGatewayConnectionType other) => string.Equals(_value, other._value, StringComparison.InvariantCultureIgnoreCase);

        /// <inheritdoc />
        [EditorBrowsable(EditorBrowsableState.Never)]
        public override int GetHashCode() => _value?.GetHashCode() ?? 0;
        /// <inheritdoc />
        public override string ToString() => _value;
    }
}
