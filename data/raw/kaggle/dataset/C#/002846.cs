// <auto-generated>
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for
// license information.
//
// Code generated by Microsoft (R) AutoRest Code Generator.
// Changes may cause incorrect behavior and will be lost if the code is
// regenerated.
// </auto-generated>

namespace Microsoft.Azure.Management.ContainerInstance
{
    using Microsoft.Rest;
    using Microsoft.Rest.Azure;
    using Models;
    using System.Collections;
    using System.Collections.Generic;
    using System.Threading;
    using System.Threading.Tasks;

    /// <summary>
    /// Extension methods for LocationOperations.
    /// </summary>
    public static partial class LocationOperationsExtensions
    {
            /// <summary>
            /// Get the usage for a subscription
            /// </summary>
            /// <param name='operations'>
            /// The operations group for this extension method.
            /// </param>
            /// <param name='location'>
            /// The identifier for the physical azure location.
            /// </param>
            public static IEnumerable<Usage> ListUsage(this ILocationOperations operations, string location)
            {
                return operations.ListUsageAsync(location).GetAwaiter().GetResult();
            }

            /// <summary>
            /// Get the usage for a subscription
            /// </summary>
            /// <param name='operations'>
            /// The operations group for this extension method.
            /// </param>
            /// <param name='location'>
            /// The identifier for the physical azure location.
            /// </param>
            /// <param name='cancellationToken'>
            /// The cancellation token.
            /// </param>
            public static async Task<IEnumerable<Usage>> ListUsageAsync(this ILocationOperations operations, string location, CancellationToken cancellationToken = default(CancellationToken))
            {
                using (var _result = await operations.ListUsageWithHttpMessagesAsync(location, null, cancellationToken).ConfigureAwait(false))
                {
                    return _result.Body;
                }
            }

            /// <summary>
            /// Get the list of cached images.
            /// </summary>
            /// <remarks>
            /// Get the list of cached images on specific OS type for a subscription in a
            /// region.
            /// </remarks>
            /// <param name='operations'>
            /// The operations group for this extension method.
            /// </param>
            /// <param name='location'>
            /// The identifier for the physical azure location.
            /// </param>
            public static IPage<CachedImages> ListCachedImages(this ILocationOperations operations, string location)
            {
                return operations.ListCachedImagesAsync(location).GetAwaiter().GetResult();
            }

            /// <summary>
            /// Get the list of cached images.
            /// </summary>
            /// <remarks>
            /// Get the list of cached images on specific OS type for a subscription in a
            /// region.
            /// </remarks>
            /// <param name='operations'>
            /// The operations group for this extension method.
            /// </param>
            /// <param name='location'>
            /// The identifier for the physical azure location.
            /// </param>
            /// <param name='cancellationToken'>
            /// The cancellation token.
            /// </param>
            public static async Task<IPage<CachedImages>> ListCachedImagesAsync(this ILocationOperations operations, string location, CancellationToken cancellationToken = default(CancellationToken))
            {
                using (var _result = await operations.ListCachedImagesWithHttpMessagesAsync(location, null, cancellationToken).ConfigureAwait(false))
                {
                    return _result.Body;
                }
            }

            /// <summary>
            /// Get the list of capabilities of the location.
            /// </summary>
            /// <remarks>
            /// Get the list of CPU/memory/GPU capabilities of a region.
            /// </remarks>
            /// <param name='operations'>
            /// The operations group for this extension method.
            /// </param>
            /// <param name='location'>
            /// The identifier for the physical azure location.
            /// </param>
            public static IPage<Capabilities> ListCapabilities(this ILocationOperations operations, string location)
            {
                return operations.ListCapabilitiesAsync(location).GetAwaiter().GetResult();
            }

            /// <summary>
            /// Get the list of capabilities of the location.
            /// </summary>
            /// <remarks>
            /// Get the list of CPU/memory/GPU capabilities of a region.
            /// </remarks>
            /// <param name='operations'>
            /// The operations group for this extension method.
            /// </param>
            /// <param name='location'>
            /// The identifier for the physical azure location.
            /// </param>
            /// <param name='cancellationToken'>
            /// The cancellation token.
            /// </param>
            public static async Task<IPage<Capabilities>> ListCapabilitiesAsync(this ILocationOperations operations, string location, CancellationToken cancellationToken = default(CancellationToken))
            {
                using (var _result = await operations.ListCapabilitiesWithHttpMessagesAsync(location, null, cancellationToken).ConfigureAwait(false))
                {
                    return _result.Body;
                }
            }

            /// <summary>
            /// Get the list of cached images.
            /// </summary>
            /// <remarks>
            /// Get the list of cached images on specific OS type for a subscription in a
            /// region.
            /// </remarks>
            /// <param name='operations'>
            /// The operations group for this extension method.
            /// </param>
            /// <param name='nextPageLink'>
            /// The NextLink from the previous successful call to List operation.
            /// </param>
            public static IPage<CachedImages> ListCachedImagesNext(this ILocationOperations operations, string nextPageLink)
            {
                return operations.ListCachedImagesNextAsync(nextPageLink).GetAwaiter().GetResult();
            }

            /// <summary>
            /// Get the list of cached images.
            /// </summary>
            /// <remarks>
            /// Get the list of cached images on specific OS type for a subscription in a
            /// region.
            /// </remarks>
            /// <param name='operations'>
            /// The operations group for this extension method.
            /// </param>
            /// <param name='nextPageLink'>
            /// The NextLink from the previous successful call to List operation.
            /// </param>
            /// <param name='cancellationToken'>
            /// The cancellation token.
            /// </param>
            public static async Task<IPage<CachedImages>> ListCachedImagesNextAsync(this ILocationOperations operations, string nextPageLink, CancellationToken cancellationToken = default(CancellationToken))
            {
                using (var _result = await operations.ListCachedImagesNextWithHttpMessagesAsync(nextPageLink, null, cancellationToken).ConfigureAwait(false))
                {
                    return _result.Body;
                }
            }

            /// <summary>
            /// Get the list of capabilities of the location.
            /// </summary>
            /// <remarks>
            /// Get the list of CPU/memory/GPU capabilities of a region.
            /// </remarks>
            /// <param name='operations'>
            /// The operations group for this extension method.
            /// </param>
            /// <param name='nextPageLink'>
            /// The NextLink from the previous successful call to List operation.
            /// </param>
            public static IPage<Capabilities> ListCapabilitiesNext(this ILocationOperations operations, string nextPageLink)
            {
                return operations.ListCapabilitiesNextAsync(nextPageLink).GetAwaiter().GetResult();
            }

            /// <summary>
            /// Get the list of capabilities of the location.
            /// </summary>
            /// <remarks>
            /// Get the list of CPU/memory/GPU capabilities of a region.
            /// </remarks>
            /// <param name='operations'>
            /// The operations group for this extension method.
            /// </param>
            /// <param name='nextPageLink'>
            /// The NextLink from the previous successful call to List operation.
            /// </param>
            /// <param name='cancellationToken'>
            /// The cancellation token.
            /// </param>
            public static async Task<IPage<Capabilities>> ListCapabilitiesNextAsync(this ILocationOperations operations, string nextPageLink, CancellationToken cancellationToken = default(CancellationToken))
            {
                using (var _result = await operations.ListCapabilitiesNextWithHttpMessagesAsync(nextPageLink, null, cancellationToken).ConfigureAwait(false))
                {
                    return _result.Body;
                }
            }

    }
}
