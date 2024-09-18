// <auto-generated>
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for
// license information.
//
// Code generated by Microsoft (R) AutoRest Code Generator.
// Changes may cause incorrect behavior and will be lost if the code is
// regenerated.
// </auto-generated>

namespace Microsoft.Azure.Management.LabServices.Models
{
    using Microsoft.Rest;
    using Newtonsoft.Json;
    using System.Linq;

    /// <summary>
    /// The base virtual machine configuration for a lab.
    /// </summary>
    public partial class VirtualMachineProfile
    {
        /// <summary>
        /// Initializes a new instance of the VirtualMachineProfile class.
        /// </summary>
        public VirtualMachineProfile()
        {
            CustomInit();
        }

        /// <summary>
        /// Initializes a new instance of the VirtualMachineProfile class.
        /// </summary>
        /// <param name="createOption">Indicates what lab virtual machines are
        /// created from. Possible values include: 'Image',
        /// 'TemplateVM'</param>
        /// <param name="imageReference">The image configuration for lab
        /// virtual machines.</param>
        /// <param name="sku">The SKU for the lab. Defines the type of virtual
        /// machines used in the lab.</param>
        /// <param name="usageQuota">The initial quota alloted to each lab
        /// user. Must be a time span between 0 and 9999 hours.</param>
        /// <param name="adminUser">Credentials for the admin user on the
        /// VM.</param>
        /// <param name="osType">The OS type of the image. Possible values
        /// include: 'Windows', 'Linux'</param>
        /// <param name="additionalCapabilities">Additional VM
        /// capabilities.</param>
        /// <param name="useSharedPassword">Enabling this option will use the
        /// same password for all user VMs. Possible values include: 'Enabled',
        /// 'Disabled'</param>
        /// <param name="nonAdminUser">Credentials for the non-admin user on
        /// the VM, if one exists.</param>
        public VirtualMachineProfile(CreateOption createOption, ImageReference imageReference, Sku sku, System.TimeSpan usageQuota, Credentials adminUser, OsType? osType = default(OsType?), VirtualMachineAdditionalCapabilities additionalCapabilities = default(VirtualMachineAdditionalCapabilities), EnableState? useSharedPassword = default(EnableState?), Credentials nonAdminUser = default(Credentials))
        {
            CreateOption = createOption;
            ImageReference = imageReference;
            OsType = osType;
            Sku = sku;
            AdditionalCapabilities = additionalCapabilities;
            UsageQuota = usageQuota;
            UseSharedPassword = useSharedPassword;
            AdminUser = adminUser;
            NonAdminUser = nonAdminUser;
            CustomInit();
        }

        /// <summary>
        /// An initialization method that performs custom operations like setting defaults
        /// </summary>
        partial void CustomInit();

        /// <summary>
        /// Gets or sets indicates what lab virtual machines are created from.
        /// Possible values include: 'Image', 'TemplateVM'
        /// </summary>
        [JsonProperty(PropertyName = "createOption")]
        public CreateOption CreateOption { get; set; }

        /// <summary>
        /// Gets or sets the image configuration for lab virtual machines.
        /// </summary>
        [JsonProperty(PropertyName = "imageReference")]
        public ImageReference ImageReference { get; set; }

        /// <summary>
        /// Gets the OS type of the image. Possible values include: 'Windows',
        /// 'Linux'
        /// </summary>
        [JsonProperty(PropertyName = "osType")]
        public OsType? OsType { get; private set; }

        /// <summary>
        /// Gets or sets the SKU for the lab. Defines the type of virtual
        /// machines used in the lab.
        /// </summary>
        [JsonProperty(PropertyName = "sku")]
        public Sku Sku { get; set; }

        /// <summary>
        /// Gets or sets additional VM capabilities.
        /// </summary>
        [JsonProperty(PropertyName = "additionalCapabilities")]
        public VirtualMachineAdditionalCapabilities AdditionalCapabilities { get; set; }

        /// <summary>
        /// Gets or sets the initial quota alloted to each lab user. Must be a
        /// time span between 0 and 9999 hours.
        /// </summary>
        [JsonProperty(PropertyName = "usageQuota")]
        public System.TimeSpan UsageQuota { get; set; }

        /// <summary>
        /// Gets or sets enabling this option will use the same password for
        /// all user VMs. Possible values include: 'Enabled', 'Disabled'
        /// </summary>
        [JsonProperty(PropertyName = "useSharedPassword")]
        public EnableState? UseSharedPassword { get; set; }

        /// <summary>
        /// Gets or sets credentials for the admin user on the VM.
        /// </summary>
        [JsonProperty(PropertyName = "adminUser")]
        public Credentials AdminUser { get; set; }

        /// <summary>
        /// Gets or sets credentials for the non-admin user on the VM, if one
        /// exists.
        /// </summary>
        [JsonProperty(PropertyName = "nonAdminUser")]
        public Credentials NonAdminUser { get; set; }

        /// <summary>
        /// Validate the object.
        /// </summary>
        /// <exception cref="ValidationException">
        /// Thrown if validation fails
        /// </exception>
        public virtual void Validate()
        {
            if (ImageReference == null)
            {
                throw new ValidationException(ValidationRules.CannotBeNull, "ImageReference");
            }
            if (Sku == null)
            {
                throw new ValidationException(ValidationRules.CannotBeNull, "Sku");
            }
            if (AdminUser == null)
            {
                throw new ValidationException(ValidationRules.CannotBeNull, "AdminUser");
            }
            if (Sku != null)
            {
                Sku.Validate();
            }
            if (AdminUser != null)
            {
                AdminUser.Validate();
            }
            if (NonAdminUser != null)
            {
                NonAdminUser.Validate();
            }
        }
    }
}
