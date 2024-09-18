// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// <auto-generated/>

#nullable disable

using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Azure;
using Azure.Core;
using Azure.ResourceManager.Network.Models;

namespace Azure.ResourceManager.Network
{
    internal class AvailableProvidersListOperationSource : IOperationSource<AvailableProvidersList>
    {
        AvailableProvidersList IOperationSource<AvailableProvidersList>.CreateResult(Response response, CancellationToken cancellationToken)
        {
            using var document = JsonDocument.Parse(response.ContentStream);
            return AvailableProvidersList.DeserializeAvailableProvidersList(document.RootElement);
        }

        async ValueTask<AvailableProvidersList> IOperationSource<AvailableProvidersList>.CreateResultAsync(Response response, CancellationToken cancellationToken)
        {
            using var document = await JsonDocument.ParseAsync(response.ContentStream, default, cancellationToken).ConfigureAwait(false);
            return AvailableProvidersList.DeserializeAvailableProvidersList(document.RootElement);
        }
    }
}
