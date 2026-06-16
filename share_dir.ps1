<#
.SYNOPSIS
    设置文件夹共享（Everyone 完全控制）
.DESCRIPTION
    为指定文件夹添加 SMB 共享，并赋予 Everyone 完全控制权限。
    默认不修改系统级安全策略；如需关闭匿名/SMB 签名限制，
    请额外指定 -DangerousMode。
.PARAMETER FolderPath
    要共享的文件夹路径（例如：D:\Scans）
.PARAMETER ShareName
    共享名称（默认使用文件夹名）
.PARAMETER DangerousMode
    同时修改系统级安全策略（restrictanonymous、SMB 签名关闭），
    以允许 Everyone 无密码访问。可能降低整机安全性，请确认后使用。
.EXAMPLE
    .\share_dir.ps1 -FolderPath "D:\Scans" -ShareName "PublicScan"
.EXAMPLE
    .\share_dir.ps1 -FolderPath "D:\scan" -DangerousMode
#>

param(
    [Parameter(Mandatory = $true)]
    [ValidateScript({Test-Path $_ -PathType Container})]
    [string]$FolderPath,

    [string]$ShareName,

    [switch]$DangerousMode
)

# 检查管理员权限
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator
)
if (-not $isAdmin) {
    Write-Host "❌ 需要以管理员身份运行此脚本。" -ForegroundColor Red
    Write-Host "   请右键 PowerShell → 以管理员身份运行，再重试。" -ForegroundColor Yellow
    exit 1
}

$ErrorActionPreference = "Stop"

if (-not $ShareName) {
    $ShareName = Split-Path $FolderPath -Leaf
}

Write-Host "=" * 52
Write-Host "  共享文件夹设置"
Write-Host "  路径：$FolderPath"
Write-Host "  共享名：$ShareName"
Write-Host "=" * 52

# 保存原始注册表值，供回滚提示用
$originalRestrictAnon = $null
$originalRestrictAnonSam = $null
$originalSMBReqSig = $null

try {
    # ---- 1. NTFS 权限 ----
    Write-Host "`n[1/3] 设置 NTFS 权限（Everyone 完全控制）..." -ForegroundColor Cyan
    # 只重置根目录（不加 /T），子目录由继承（OI)(CI）自动应用
    icacls $FolderPath /reset /Q
    icacls $FolderPath /grant "Everyone:(OI)(CI)F" /Q
    Write-Host "  ✓ NTFS 权限已设置" -ForegroundColor Green

    # ---- 2. 共享 ----
    if (Get-SmbShare -Name $ShareName -ErrorAction SilentlyContinue) {
        Remove-SmbShare -Name $ShareName -Force
        Write-Host "  ✓ 已移除旧的同名共享" -ForegroundColor Yellow
    }

    Write-Host "[2/3] 创建共享：$ShareName ..." -ForegroundColor Cyan
    New-SmbShare -Name $ShareName -Path $FolderPath -FullAccess "Everyone" `
        -Description "一键共享（Everyone 完全控制）" -Force
    Write-Host "  ✓ 共享已创建" -ForegroundColor Green

    # ---- 3. 防火墙 ----
    Write-Host "[3/3] 放行防火墙 ..." -ForegroundColor Cyan
    Set-NetFirewallRule -DisplayGroup "文件和打印机共享" -Enabled True -Profile Any
    Write-Host "  ✓ 防火墙已放行" -ForegroundColor Green

    # ---- 4. 系统级安全策略（可选）----
    if ($DangerousMode) {
        Write-Host "`n⚠️  启用 DangerousMode：修改系统安全策略..." -ForegroundColor Yellow
        $regPath = "HKLM:\SYSTEM\CurrentControlSet\Control\Lsa"
        $originalRestrictAnon = Get-ItemProperty -Path $regPath -Name "restrictanonymous" -ErrorAction SilentlyContinue
        $originalRestrictAnonSam = Get-ItemProperty -Path $regPath -Name "restrictanonymoussam" -ErrorAction SilentlyContinue
        Set-ItemProperty -Path $regPath -Name "restrictanonymous" -Value 0 -Type DWord -Force
        Set-ItemProperty -Path $regPath -Name "restrictanonymoussam" -Value 0 -Type DWord -Force
        $originalSMBReqSig = (Get-SmbClientConfiguration).RequireSecuritySignature
        Set-SmbClientConfiguration -RequireSecuritySignature $false -Force -Confirm:$false | Out-Null
        Write-Host "  ✓ restrictanonymous = 0" -ForegroundColor Green
        Write-Host "  ✓ restrictanonymoussam = 0" -ForegroundColor Green
        Write-Host "  ✓ RequireSecuritySignature = false" -ForegroundColor Green
    }

    Write-Host "`n✅ 共享设置完成！" -ForegroundColor Green
    Write-Host "  共享路径：\\$env:COMPUTERNAME\$ShareName" -ForegroundColor Cyan
    Write-Host "  共享位置：$FolderPath" -ForegroundColor Gray
    Get-SmbShare -Name $ShareName | Format-List

    # ---- 回滚提示 ----
    Write-Host "`n--- 回滚信息（请保存）---" -ForegroundColor Magenta
    Write-Host "  撤销 NTFS 权限：icacls `"$FolderPath`" /reset /T" -ForegroundColor Gray
    Write-Host "  删除共享：Remove-SmbShare -Name `"$ShareName`" -Force" -ForegroundColor Gray
    if ($DangerousMode) {
        if ($originalRestrictAnon) {
            Write-Host "  恢复 restrictanonymous：Set-ItemProperty -Path `"$regPath`" -Name restrictanonymous -Value $($originalRestrictAnon.restrictanonymous) -Type DWord" -ForegroundColor Gray
        }
        if ($originalRestrictAnonSam) {
            Write-Host "  恢复 restrictanonymoussam：Set-ItemProperty -Path `"$regPath`" -Name restrictanonymoussam -Value $($originalRestrictAnonSam.restrictanonymoussam) -Type DWord" -ForegroundColor Gray
        }
        if ($null -ne $originalSMBReqSig) {
            Write-Host "  恢复 SMB 签名：Set-SmbClientConfiguration -RequireSecuritySignature `$$originalSMBReqSig -Force" -ForegroundColor Gray
        }
    }

} catch {
    Write-Host "`n❌ 操作失败：" -ForegroundColor Red
    Write-Host "   $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
