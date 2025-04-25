const fs = require('fs')
const {execSync} = require('child_process')

// 交换路径
const inpath = 'H:\\temp\\个人备份\\李金科\\exchangeModel\\input'
const outpath = 'H:\\temp\\个人备份\\李金科\\exchangeModel\\output'

// 程序运行环境
const env = ['client','server'][1]
console.log('启动环境：',env);

// 监听地址
const input = inpath
const output = outpath

// 监听器实现
let ints = 0
function scan(){
  let inflag = false

  // 要监控的地址
  let innow = fs.statSync(input).mtimeMs

  // 文件改动判断
  if(ints === 0) ints = innow
  else if(ints !== innow) inflag = true

  // 监听到输入改变
  if(inflag){
    ints = innow
    
    try {
      const result = execSync(`py ${input}`).toString()
      console.log('服务端输入文件执行成功')
      fs.writeFileSync(output,result)
      console.log('服务端输出文件写入成功')
    } catch (error) {
      console.error(error)
    }
  }
}
setInterval(scan,400)
