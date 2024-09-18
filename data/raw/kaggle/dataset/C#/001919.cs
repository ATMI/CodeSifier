// <auto-generated>
// Code generated by Microsoft (R) AutoRest Code Generator.
// Changes may cause incorrect behavior and will be lost if the code is
// regenerated.
// </auto-generated>

namespace Microsoft.Azure.CognitiveServices.Search.AutoSuggest
{
    using Microsoft.Rest;
    using Microsoft.Rest.Serialization;
    using Models;
    using Newtonsoft.Json;
    using System.Collections;
    using System.Collections.Generic;
    using System.Net;
    using System.Net.Http;
    using System.Threading;
    using System.Threading.Tasks;

    /// <summary>
    /// The AutoSuggest Search API lets you send a search query to Bing and get
    /// back a list of news that are relevant to the search query. This section
    /// provides technical details about the query parameters and headers that
    /// you use to request news and the JSON response objects that contain
    /// them. For examples that show how to make requests, see [Searching the
    /// web for
    /// AutoSuggest](https://docs.microsoft.com/en-us/rest/api/cognitiveservices/bing-autosuggest-api-v7-reference).
    /// </summary>
    public partial class AutoSuggestClient : ServiceClient<AutoSuggestClient>, IAutoSuggestClient
    {
        /// <summary>
        /// The base URI of the service.
        /// </summary>
        internal string BaseUri {get; set;}

        /// <summary>
        /// Gets or sets json serialization settings.
        /// </summary>
        public JsonSerializerSettings SerializationSettings { get; private set; }

        /// <summary>
        /// Gets or sets json deserialization settings.
        /// </summary>
        public JsonSerializerSettings DeserializationSettings { get; private set; }

        /// <summary>
        /// Supported Cognitive Services endpoints (protocol and hostname, for example:
        /// "https://westus.api.cognitive.microsoft.com",
        /// "https://api.cognitive.microsoft.com").
        /// </summary>
        public string Endpoint { get; set; }

        /// <summary>
        /// Subscription credentials which uniquely identify client subscription.
        /// </summary>
        public ServiceClientCredentials Credentials { get; private set; }

        /// <summary>
        /// Initializes a new instance of the AutoSuggestClient class.
        /// </summary>
        /// <param name='httpClient'>
        /// HttpClient to be used
        /// </param>
        /// <param name='disposeHttpClient'>
        /// True: will dispose the provided httpClient on calling AutoSuggestClient.Dispose(). False: will not dispose provided httpClient</param>
        protected AutoSuggestClient(HttpClient httpClient, bool disposeHttpClient) : base(httpClient, disposeHttpClient)
        {
            Initialize();
        }

        /// <summary>
        /// Initializes a new instance of the AutoSuggestClient class.
        /// </summary>
        /// <param name='handlers'>
        /// Optional. The delegating handlers to add to the http client pipeline.
        /// </param>
        protected AutoSuggestClient(params DelegatingHandler[] handlers) : base(handlers)
        {
            Initialize();
        }

        /// <summary>
        /// Initializes a new instance of the AutoSuggestClient class.
        /// </summary>
        /// <param name='rootHandler'>
        /// Optional. The http client handler used to handle http transport.
        /// </param>
        /// <param name='handlers'>
        /// Optional. The delegating handlers to add to the http client pipeline.
        /// </param>
        protected AutoSuggestClient(HttpClientHandler rootHandler, params DelegatingHandler[] handlers) : base(rootHandler, handlers)
        {
            Initialize();
        }

        /// <summary>
        /// Initializes a new instance of the AutoSuggestClient class.
        /// </summary>
        /// <param name='credentials'>
        /// Required. Subscription credentials which uniquely identify client subscription.
        /// </param>
        /// <param name='handlers'>
        /// Optional. The delegating handlers to add to the http client pipeline.
        /// </param>
        /// <exception cref="System.ArgumentNullException">
        /// Thrown when a required parameter is null
        /// </exception>
        public AutoSuggestClient(ServiceClientCredentials credentials, params DelegatingHandler[] handlers) : this(handlers)
        {
            if (credentials == null)
            {
                throw new System.ArgumentNullException("credentials");
            }
            Credentials = credentials;
            if (Credentials != null)
            {
                Credentials.InitializeServiceClient(this);
            }
        }

        /// <summary>
        /// Initializes a new instance of the AutoSuggestClient class.
        /// </summary>
        /// <param name='credentials'>
        /// Required. Subscription credentials which uniquely identify client subscription.
        /// </param>
        /// <param name='httpClient'>
        /// HttpClient to be used
        /// </param>
        /// <param name='disposeHttpClient'>
        /// True: will dispose the provided httpClient on calling AutoSuggestClient.Dispose(). False: will not dispose provided httpClient</param>
        /// <exception cref="System.ArgumentNullException">
        /// Thrown when a required parameter is null
        /// </exception>
        public AutoSuggestClient(ServiceClientCredentials credentials, HttpClient httpClient, bool disposeHttpClient) : this(httpClient, disposeHttpClient)
        {
            if (credentials == null)
            {
                throw new System.ArgumentNullException("credentials");
            }
            Credentials = credentials;
            if (Credentials != null)
            {
                Credentials.InitializeServiceClient(this);
            }
        }

        /// <summary>
        /// Initializes a new instance of the AutoSuggestClient class.
        /// </summary>
        /// <param name='credentials'>
        /// Required. Subscription credentials which uniquely identify client subscription.
        /// </param>
        /// <param name='rootHandler'>
        /// Optional. The http client handler used to handle http transport.
        /// </param>
        /// <param name='handlers'>
        /// Optional. The delegating handlers to add to the http client pipeline.
        /// </param>
        /// <exception cref="System.ArgumentNullException">
        /// Thrown when a required parameter is null
        /// </exception>
        public AutoSuggestClient(ServiceClientCredentials credentials, HttpClientHandler rootHandler, params DelegatingHandler[] handlers) : this(rootHandler, handlers)
        {
            if (credentials == null)
            {
                throw new System.ArgumentNullException("credentials");
            }
            Credentials = credentials;
            if (Credentials != null)
            {
                Credentials.InitializeServiceClient(this);
            }
        }

        /// <summary>
        /// An optional partial-method to perform custom initialization.
        ///</summary>
        partial void CustomInitialize();
        /// <summary>
        /// Initializes client properties.
        /// </summary>
        private void Initialize()
        {
            BaseUri = "{Endpoint}/bing/v7.0";
            Endpoint = "https://api.cognitive.microsoft.com";
            SerializationSettings = new JsonSerializerSettings
            {
                Formatting = Newtonsoft.Json.Formatting.Indented,
                DateFormatHandling = Newtonsoft.Json.DateFormatHandling.IsoDateFormat,
                DateTimeZoneHandling = Newtonsoft.Json.DateTimeZoneHandling.Utc,
                NullValueHandling = Newtonsoft.Json.NullValueHandling.Ignore,
                ReferenceLoopHandling = Newtonsoft.Json.ReferenceLoopHandling.Serialize,
                ContractResolver = new ReadOnlyJsonContractResolver(),
                Converters = new  List<JsonConverter>
                    {
                        new Iso8601TimeSpanConverter()
                    }
            };
            DeserializationSettings = new JsonSerializerSettings
            {
                DateFormatHandling = Newtonsoft.Json.DateFormatHandling.IsoDateFormat,
                DateTimeZoneHandling = Newtonsoft.Json.DateTimeZoneHandling.Utc,
                NullValueHandling = Newtonsoft.Json.NullValueHandling.Ignore,
                ReferenceLoopHandling = Newtonsoft.Json.ReferenceLoopHandling.Serialize,
                ContractResolver = new ReadOnlyJsonContractResolver(),
                Converters = new List<JsonConverter>
                    {
                        new Iso8601TimeSpanConverter()
                    }
            };
            SerializationSettings.Converters.Add(new PolymorphicSerializeJsonConverter<SuggestionsSuggestionGroup>("_type"));
            DeserializationSettings.Converters.Add(new  PolymorphicDeserializeJsonConverter<SuggestionsSuggestionGroup>("_type"));
            SerializationSettings.Converters.Add(new PolymorphicSerializeJsonConverter<QueryContext>("_type"));
            DeserializationSettings.Converters.Add(new  PolymorphicDeserializeJsonConverter<QueryContext>("_type"));
            SerializationSettings.Converters.Add(new PolymorphicSerializeJsonConverter<Error>("_type"));
            DeserializationSettings.Converters.Add(new  PolymorphicDeserializeJsonConverter<Error>("_type"));
            SerializationSettings.Converters.Add(new PolymorphicSerializeJsonConverter<ResponseBase>("_type"));
            DeserializationSettings.Converters.Add(new  PolymorphicDeserializeJsonConverter<ResponseBase>("_type"));
            CustomInitialize();
        }
        /// <summary>
        /// The AutoSuggest API lets you send a search query to Bing and get back a
        /// list of suggestions. This section provides technical details about the
        /// query parameters and headers that you use to request suggestions and the
        /// JSON response objects that contain them.
        /// </summary>
        /// <param name='query'>
        /// The user's search term.
        /// </param>
        /// <param name='acceptLanguage'>
        /// A comma-delimited list of one or more languages to use for user interface
        /// strings. The list is in decreasing order of preference. For additional
        /// information, including expected format, see
        /// [RFC2616](http://www.w3.org/Protocols/rfc2616/rfc2616-sec14.html). This
        /// header and the setLang query parameter are mutually exclusive; do not
        /// specify both. If you set this header, you must also specify the
        /// [cc](https://docs.microsoft.com/en-us/rest/api/cognitiveservices/bing-autosuggest-api-v7-reference#cc)
        /// query parameter. To determine the market to return results for, Bing uses
        /// the first supported language it finds from the list and combines it with
        /// the cc parameter value. If the list does not include a supported language,
        /// Bing finds the closest language and market that supports the request or it
        /// uses an aggregated or default market for the results. To determine the
        /// market that Bing used, see the BingAPIs-Market header. Use this header and
        /// the cc query parameter only if you specify multiple languages. Otherwise,
        /// use the
        /// [mkt](https://docs.microsoft.com/en-us/rest/api/cognitiveservices/bing-autosuggest-api-v7-reference#mkt)
        /// and
        /// [setLang](https://docs.microsoft.com/en-us/rest/api/cognitiveservices/bing-autosuggest-api-v7-reference#setlang)
        /// query parameters. A user interface string is a string that's used as a
        /// label in a user interface. There are few user interface strings in the JSON
        /// response objects. Any links to Bing.com properties in the response objects
        /// apply the specified language.
        /// </param>
        /// <param name='pragma'>
        /// By default, Bing returns cached content, if available. To prevent Bing from
        /// returning cached content, set the Pragma header to no-cache (for example,
        /// Pragma: no-cache).
        /// </param>
        /// <param name='userAgent'>
        /// The user agent originating the request. Bing uses the user agent to provide
        /// mobile users with an optimized experience. Although optional, you are
        /// encouraged to always specify this header. The user-agent should be the same
        /// string that any commonly used browser sends. For information about user
        /// agents, see [RFC
        /// 2616](http://www.w3.org/Protocols/rfc2616/rfc2616-sec14.html). The
        /// following are examples of user-agent strings. Windows Phone: Mozilla/5.0
        /// (compatible; MSIE 10.0; Windows Phone 8.0; Trident/6.0; IEMobile/10.0; ARM;
        /// Touch; NOKIA; Lumia 822). Android: Mozilla / 5.0 (Linux; U; Android 2.3.5;
        /// en - us; SCH - I500 Build / GINGERBREAD) AppleWebKit / 533.1 (KHTML; like
        /// Gecko) Version / 4.0 Mobile Safari / 533.1. iPhone: Mozilla / 5.0 (iPhone;
        /// CPU iPhone OS 6_1 like Mac OS X) AppleWebKit / 536.26 (KHTML; like Gecko)
        /// Mobile / 10B142 iPhone4; 1 BingWeb / 3.03.1428.20120423. PC: Mozilla / 5.0
        /// (Windows NT 6.3; WOW64; Trident / 7.0; Touch; rv:11.0) like Gecko. iPad:
        /// Mozilla / 5.0 (iPad; CPU OS 7_0 like Mac OS X) AppleWebKit / 537.51.1
        /// (KHTML, like Gecko) Version / 7.0 Mobile / 11A465 Safari / 9537.53
        /// </param>
        /// <param name='clientId'>
        /// Bing uses this header to provide users with consistent behavior across Bing
        /// API calls. Bing often flights new features and improvements, and it uses
        /// the client ID as a key for assigning traffic on different flights. If you
        /// do not use the same client ID for a user across multiple requests, then
        /// Bing may assign the user to multiple conflicting flights. Being assigned to
        /// multiple conflicting flights can lead to an inconsistent user experience.
        /// For example, if the second request has a different flight assignment than
        /// the first, the experience may be unexpected. Also, Bing can use the client
        /// ID to tailor web results to that client ID’s search history, providing a
        /// richer experience for the user. Bing also uses this header to help improve
        /// result rankings by analyzing the activity generated by a client ID. The
        /// relevance improvements help with better quality of results delivered by
        /// Bing APIs and in turn enables higher click-through rates for the API
        /// consumer. IMPORTANT: Although optional, you should consider this header
        /// required. Persisting the client ID across multiple requests for the same
        /// end user and device combination enables 1) the API consumer to receive a
        /// consistent user experience, and 2) higher click-through rates via better
        /// quality of results from the Bing APIs. Each user that uses your application
        /// on the device must have a unique, Bing generated client ID. If you do not
        /// include this header in the request, Bing generates an ID and returns it in
        /// the X-MSEdge-ClientID response header. The only time that you should NOT
        /// include this header in a request is the first time the user uses your app
        /// on that device. Use the client ID for each Bing API request that your app
        /// makes for this user on the device. Persist the client ID. To persist the ID
        /// in a browser app, use a persistent HTTP cookie to ensure the ID is used
        /// across all sessions. Do not use a session cookie. For other apps such as
        /// mobile apps, use the device's persistent storage to persist the ID. The
        /// next time the user uses your app on that device, get the client ID that you
        /// persisted. Bing responses may or may not include this header. If the
        /// response includes this header, capture the client ID and use it for all
        /// subsequent Bing requests for the user on that device. If you include the
        /// X-MSEdge-ClientID, you must not include cookies in the request.
        /// </param>
        /// <param name='clientIp'>
        /// The IPv4 or IPv6 address of the client device. The IP address is used to
        /// discover the user's location. Bing uses the location information to
        /// determine safe search behavior. Although optional, you are encouraged to
        /// always specify this header and the X-Search-Location header. Do not
        /// obfuscate the address (for example, by changing the last octet to 0).
        /// Obfuscating the address results in the location not being anywhere near the
        /// device's actual location, which may result in Bing serving erroneous
        /// results.
        /// </param>
        /// <param name='location'>
        /// A semicolon-delimited list of key/value pairs that describe the client's
        /// geographical location. Bing uses the location information to determine safe
        /// search behavior and to return relevant local content. Specify the key/value
        /// pair as &lt;key&gt;:&lt;value&gt;. The following are the keys that you use
        /// to specify the user's location. lat (required): The latitude of the
        /// client's location, in degrees. The latitude must be greater than or equal
        /// to -90.0 and less than or equal to +90.0. Negative values indicate southern
        /// latitudes and positive values indicate northern latitudes. long (required):
        /// The longitude of the client's location, in degrees. The longitude must be
        /// greater than or equal to -180.0 and less than or equal to +180.0. Negative
        /// values indicate western longitudes and positive values indicate eastern
        /// longitudes. re (required): The radius, in meters, which specifies the
        /// horizontal accuracy of the coordinates. Pass the value returned by the
        /// device's location service. Typical values might be 22m for GPS/Wi-Fi, 380m
        /// for cell tower triangulation, and 18,000m for reverse IP lookup. ts
        /// (optional): The UTC UNIX timestamp of when the client was at the location.
        /// (The UNIX timestamp is the number of seconds since January 1, 1970.) head
        /// (optional): The client's relative heading or direction of travel. Specify
        /// the direction of travel as degrees from 0 through 360, counting clockwise
        /// relative to true north. Specify this key only if the sp key is nonzero. sp
        /// (optional): The horizontal velocity (speed), in meters per second, that the
        /// client device is traveling. alt (optional): The altitude of the client
        /// device, in meters. are (optional): The radius, in meters, that specifies
        /// the vertical accuracy of the coordinates. Specify this key only if you
        /// specify the alt key. Although many of the keys are optional, the more
        /// information that you provide, the more accurate the location results are.
        /// Although optional, you are encouraged to always specify the user's
        /// geographical location. Providing the location is especially important if
        /// the client's IP address does not accurately reflect the user's physical
        /// location (for example, if the client uses VPN). For optimal results, you
        /// should include this header and the X-MSEdge-ClientIP header, but at a
        /// minimum, you should include this header.
        /// </param>
        /// <param name='countryCode'>
        /// A 2-character country code of the country where the results come from. This
        /// API supports only the United States market. If you specify this query
        /// parameter, it must be set to us. If you set this parameter, you must also
        /// specify the Accept-Language header. Bing uses the first supported language
        /// it finds from the languages list, and combine that language with the
        /// country code that you specify to determine the market to return results
        /// for. If the languages list does not include a supported language, Bing
        /// finds the closest language and market that supports the request, or it may
        /// use an aggregated or default market for the results instead of a specified
        /// one. You should use this query parameter and the Accept-Language query
        /// parameter only if you specify multiple languages; otherwise, you should use
        /// the mkt and setLang query parameters. This parameter and the mkt query
        /// parameter are mutually exclusive—do not specify both.
        /// </param>
        /// <param name='market'>
        /// The market where the results come from. You are strongly encouraged to
        /// always specify the market, if known. Specifying the market helps Bing route
        /// the request and return an appropriate and optimal response. This parameter
        /// and the cc query parameter are mutually exclusive—do not specify both.
        /// </param>
        /// <param name='safeSearch'>
        /// Filter suggestions for adult content. The following are the possible filter
        /// values. Off: Return suggestions with adult text, images, or videos.
        /// Moderate: Return suggestion with adult text but not adult images or videos.
        /// Strict: Do not return news articles with adult text, images, or videos. If
        /// the request comes from a market that Bing's adult policy requires that
        /// safeSearch is set to Strict, Bing ignores the safeSearch value and uses
        /// Strict. If you use the site: query operator, there is the chance that the
        /// response may contain adult content regardless of what the safeSearch query
        /// parameter is set to. Use site: only if you are aware of the content on the
        /// site and your scenario supports the possibility of adult content. Possible
        /// values include: 'Off', 'Moderate', 'Strict'
        /// </param>
        /// <param name='setLang'>
        /// The language to use for user interface strings. Specify the language using
        /// the ISO 639-1 2-letter language code. For example, the language code for
        /// English is EN. The default is EN (English). Although optional, you should
        /// always specify the language. Typically, you set setLang to the same
        /// language specified by mkt unless the user wants the user interface strings
        /// displayed in a different language. This parameter and the Accept-Language
        /// header are mutually exclusive; do not specify both. A user interface string
        /// is a string that's used as a label in a user interface. There are few user
        /// interface strings in the JSON response objects. Also, any links to Bing.com
        /// properties in the response objects apply the specified language.
        /// </param>
        /// <param name='responseFormat'>
        /// The media type to use for the response. The following are the possible
        /// case-insensitive values: JSON, JSONLD. The default is JSON. If you specify
        /// JSONLD, the response body includes JSON-LD objects that contain the search
        /// results.
        /// </param>
        /// <param name='customHeaders'>
        /// Headers that will be added to request.
        /// </param>
        /// <param name='cancellationToken'>
        /// The cancellation token.
        /// </param>
        /// <exception cref="ErrorResponseException">
        /// Thrown when the operation returned an invalid status code
        /// </exception>
        /// <exception cref="SerializationException">
        /// Thrown when unable to deserialize the response
        /// </exception>
        /// <exception cref="ValidationException">
        /// Thrown when a required parameter is null
        /// </exception>
        /// <exception cref="System.ArgumentNullException">
        /// Thrown when a required parameter is null
        /// </exception>
        /// <return>
        /// A response object containing the response body and response headers.
        /// </return>
        public async Task<HttpOperationResponse<Suggestions>> AutoSuggestMethodWithHttpMessagesAsync(string query, string acceptLanguage = default(string), string pragma = default(string), string userAgent = default(string), string clientId = default(string), string clientIp = default(string), string location = default(string), string countryCode = default(string), string market = "en-us", string safeSearch = default(string), string setLang = default(string), IList<string> responseFormat = default(IList<string>), Dictionary<string, List<string>> customHeaders = null, CancellationToken cancellationToken = default(CancellationToken))
        {
            if (Endpoint == null)
            {
                throw new ValidationException(ValidationRules.CannotBeNull, "this.Endpoint");
            }
            if (query == null)
            {
                throw new ValidationException(ValidationRules.CannotBeNull, "query");
            }
            string xBingApisSDK = "true";
            // Tracing
            bool _shouldTrace = ServiceClientTracing.IsEnabled;
            string _invocationId = null;
            if (_shouldTrace)
            {
                _invocationId = ServiceClientTracing.NextInvocationId.ToString();
                Dictionary<string, object> tracingParameters = new Dictionary<string, object>();
                tracingParameters.Add("xBingApisSDK", xBingApisSDK);
                tracingParameters.Add("acceptLanguage", acceptLanguage);
                tracingParameters.Add("pragma", pragma);
                tracingParameters.Add("userAgent", userAgent);
                tracingParameters.Add("clientId", clientId);
                tracingParameters.Add("clientIp", clientIp);
                tracingParameters.Add("location", location);
                tracingParameters.Add("countryCode", countryCode);
                tracingParameters.Add("market", market);
                tracingParameters.Add("query", query);
                tracingParameters.Add("safeSearch", safeSearch);
                tracingParameters.Add("setLang", setLang);
                tracingParameters.Add("responseFormat", responseFormat);
                tracingParameters.Add("cancellationToken", cancellationToken);
                ServiceClientTracing.Enter(_invocationId, this, "AutoSuggestMethod", tracingParameters);
            }
            // Construct URL
            var _baseUrl = BaseUri;
            var _url = _baseUrl + (_baseUrl.EndsWith("/") ? "" : "/") + "Suggestions";
            _url = _url.Replace("{Endpoint}", Endpoint);
            List<string> _queryParameters = new List<string>();
            if (countryCode != null)
            {
                _queryParameters.Add(string.Format("cc={0}", System.Uri.EscapeDataString(countryCode)));
            }
            if (market != null)
            {
                _queryParameters.Add(string.Format("mkt={0}", System.Uri.EscapeDataString(market)));
            }
            if (query != null)
            {
                _queryParameters.Add(string.Format("q={0}", System.Uri.EscapeDataString(query)));
            }
            if (safeSearch != null)
            {
                _queryParameters.Add(string.Format("safeSearch={0}", System.Uri.EscapeDataString(safeSearch)));
            }
            if (setLang != null)
            {
                _queryParameters.Add(string.Format("setLang={0}", System.Uri.EscapeDataString(setLang)));
            }
            if (responseFormat != null)
            {
                _queryParameters.Add(string.Format("ResponseFormat={0}", System.Uri.EscapeDataString(string.Join(",", responseFormat))));
            }
            if (_queryParameters.Count > 0)
            {
                _url += "?" + string.Join("&", _queryParameters);
            }
            // Create HTTP transport objects
            var _httpRequest = new HttpRequestMessage();
            HttpResponseMessage _httpResponse = null;
            _httpRequest.Method = new HttpMethod("GET");
            _httpRequest.RequestUri = new System.Uri(_url);
            // Set Headers
            if (xBingApisSDK != null)
            {
                if (_httpRequest.Headers.Contains("X-BingApis-SDK"))
                {
                    _httpRequest.Headers.Remove("X-BingApis-SDK");
                }
                _httpRequest.Headers.TryAddWithoutValidation("X-BingApis-SDK", xBingApisSDK);
            }
            if (acceptLanguage != null)
            {
                if (_httpRequest.Headers.Contains("Accept-Language"))
                {
                    _httpRequest.Headers.Remove("Accept-Language");
                }
                _httpRequest.Headers.TryAddWithoutValidation("Accept-Language", acceptLanguage);
            }
            if (pragma != null)
            {
                if (_httpRequest.Headers.Contains("Pragma"))
                {
                    _httpRequest.Headers.Remove("Pragma");
                }
                _httpRequest.Headers.TryAddWithoutValidation("Pragma", pragma);
            }
            if (userAgent != null)
            {
                if (_httpRequest.Headers.Contains("User-Agent"))
                {
                    _httpRequest.Headers.Remove("User-Agent");
                }
                _httpRequest.Headers.TryAddWithoutValidation("User-Agent", userAgent);
            }
            if (clientId != null)
            {
                if (_httpRequest.Headers.Contains("X-MSEdge-ClientID"))
                {
                    _httpRequest.Headers.Remove("X-MSEdge-ClientID");
                }
                _httpRequest.Headers.TryAddWithoutValidation("X-MSEdge-ClientID", clientId);
            }
            if (clientIp != null)
            {
                if (_httpRequest.Headers.Contains("X-MSEdge-ClientIP"))
                {
                    _httpRequest.Headers.Remove("X-MSEdge-ClientIP");
                }
                _httpRequest.Headers.TryAddWithoutValidation("X-MSEdge-ClientIP", clientIp);
            }
            if (location != null)
            {
                if (_httpRequest.Headers.Contains("X-Search-Location"))
                {
                    _httpRequest.Headers.Remove("X-Search-Location");
                }
                _httpRequest.Headers.TryAddWithoutValidation("X-Search-Location", location);
            }


            if (customHeaders != null)
            {
                foreach(var _header in customHeaders)
                {
                    if (_httpRequest.Headers.Contains(_header.Key))
                    {
                        _httpRequest.Headers.Remove(_header.Key);
                    }
                    _httpRequest.Headers.TryAddWithoutValidation(_header.Key, _header.Value);
                }
            }

            // Serialize Request
            string _requestContent = null;
            // Set Credentials
            if (Credentials != null)
            {
                cancellationToken.ThrowIfCancellationRequested();
                await Credentials.ProcessHttpRequestAsync(_httpRequest, cancellationToken).ConfigureAwait(false);
            }
            // Send Request
            if (_shouldTrace)
            {
                ServiceClientTracing.SendRequest(_invocationId, _httpRequest);
            }
            cancellationToken.ThrowIfCancellationRequested();
            _httpResponse = await HttpClient.SendAsync(_httpRequest, cancellationToken).ConfigureAwait(false);
            if (_shouldTrace)
            {
                ServiceClientTracing.ReceiveResponse(_invocationId, _httpResponse);
            }
            HttpStatusCode _statusCode = _httpResponse.StatusCode;
            cancellationToken.ThrowIfCancellationRequested();
            string _responseContent = null;
            if ((int)_statusCode != 200)
            {
                var ex = new ErrorResponseException(string.Format("Operation returned an invalid status code '{0}'", _statusCode));
                try
                {
                    _responseContent = await _httpResponse.Content.ReadAsStringAsync().ConfigureAwait(false);
                    ErrorResponse _errorBody =  SafeJsonConvert.DeserializeObject<ErrorResponse>(_responseContent, DeserializationSettings);
                    if (_errorBody != null)
                    {
                        ex.Body = _errorBody;
                    }
                }
                catch (JsonException)
                {
                    // Ignore the exception
                }
                ex.Request = new HttpRequestMessageWrapper(_httpRequest, _requestContent);
                ex.Response = new HttpResponseMessageWrapper(_httpResponse, _responseContent);
                if (_shouldTrace)
                {
                    ServiceClientTracing.Error(_invocationId, ex);
                }
                _httpRequest.Dispose();
                if (_httpResponse != null)
                {
                    _httpResponse.Dispose();
                }
                throw ex;
            }
            // Create Result
            var _result = new HttpOperationResponse<Suggestions>();
            _result.Request = _httpRequest;
            _result.Response = _httpResponse;
            // Deserialize Response
            if ((int)_statusCode == 200)
            {
                _responseContent = await _httpResponse.Content.ReadAsStringAsync().ConfigureAwait(false);
                try
                {
                    _result.Body = SafeJsonConvert.DeserializeObject<Suggestions>(_responseContent, DeserializationSettings);
                }
                catch (JsonException ex)
                {
                    _httpRequest.Dispose();
                    if (_httpResponse != null)
                    {
                        _httpResponse.Dispose();
                    }
                    throw new SerializationException("Unable to deserialize the response.", _responseContent, ex);
                }
            }
            if (_shouldTrace)
            {
                ServiceClientTracing.Exit(_invocationId, _result);
            }
            return _result;
        }

    }
}
